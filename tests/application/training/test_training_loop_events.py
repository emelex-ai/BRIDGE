"""``train_steps`` is the seam: one optimizer step per slice, one record per step.

The library used to ship the loop as well, and this file used to assert its arithmetic:
epochs, a validation pass per epoch, a merged epoch aggregate, a ``num_epochs`` override.
All of that was experiment policy and moved to the caller (docs/decisions/0013), so what
is left to pin is narrow and should stay narrow: the stream is one event per slice, in
order, carrying enough to act on, and abandoning it does no further work.
"""

import math

import pytest

from tests.conftest import batch_slices

SLICES = 4


@pytest.fixture
def pipeline(words_dataset, make_pipeline):
    return make_pipeline(words_dataset)


@pytest.fixture
def slices(words_dataset):
    return batch_slices(words_dataset, size=8)[:SLICES]


def test_one_event_per_slice_in_order(pipeline, words_dataset, slices):
    """The arithmetic of the stream, counted against the slices handed in."""
    events = list(pipeline.train_steps(words_dataset, slices, epoch=0))

    assert len(events) == SLICES
    assert [e.step for e in events] == list(range(SLICES))
    assert {e.phase for e in events} == {"train"}
    assert {e.epoch for e in events} == {0}


def test_the_epoch_index_is_the_callers_to_set(pipeline, words_dataset, slices):
    """The library counts steps within a call and nothing else. Which epoch this is, and
    how many there are, is the caller's bookkeeping."""
    events = list(pipeline.train_steps(words_dataset, slices, epoch=7))
    assert {e.epoch for e in events} == {7}
    assert [e.step for e in events] == list(range(SLICES))


def test_events_carry_a_finite_loss_and_the_words_they_trained_on(pipeline, words_dataset, slices):
    """A step event has to say enough to act on, or the caller cannot own the loop."""
    for event in pipeline.train_steps(words_dataset, slices, epoch=0):
        assert math.isfinite(float(event.metrics["loss"]))
        assert isinstance(event.metrics["word"], str)
        assert event.metrics["word"].startswith("[")


def test_the_reported_loss_carries_no_autograd_graph(pipeline, words_dataset, slices):
    """A caller keeping the stream must not be keeping the graph with it.

    Oracle: the invariant that a detached tensor has no ``grad_fn``. Handing these out
    live measured +527 MB per epoch for a caller retaining events to plot a curve.
    """
    for event in pipeline.train_steps(words_dataset, slices, epoch=0):
        loss = event.metrics["loss"]
        assert loss.grad_fn is None and not loss.requires_grad


def test_metrics_are_off_unless_asked_for(pipeline, words_dataset, slices):
    """Scoring costs ~7 ms a step, so it is opt-in per call rather than always on."""
    plain = next(iter(pipeline.train_steps(words_dataset, slices, epoch=0)))
    scored = next(
        iter(pipeline.train_steps(words_dataset, slices, epoch=0, calculate_metrics=True))
    )

    assert not any("accuracy" in key for key in plain.metrics)
    assert any("accuracy" in key for key in scored.metrics)


def test_abandoning_the_generator_does_no_further_work(pipeline, words_dataset, slices):
    """The point of a generator: stopping early must not run the rest of the steps.

    Oracle: a differential on the parameters. Taking one step of four and stopping must
    leave the model exactly where one step leaves it, which is what makes early stopping
    and step-count checkpointing possible at all.
    """
    import torch

    stream = pipeline.train_steps(words_dataset, slices, epoch=0)
    next(stream)
    after_one = pipeline.model.global_embedding.detach().clone()
    stream.close()

    assert torch.equal(after_one, pipeline.model.global_embedding)
