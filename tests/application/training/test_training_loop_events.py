"""``train_steps`` is the seam: one optimizer step per slice, one record per step.

The library used to ship the loop as well, and this file used to assert its arithmetic:
epochs, a validation pass per epoch, a merged epoch aggregate, a ``num_epochs`` override.
All of that was experiment policy and moved to the caller (docs/decisions/0013), so what
is left to pin is narrow and should stay narrow: the stream is one event per slice, in
order, carrying enough to act on, and abandoning it does no further work.
"""

import json
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


def test_events_carry_a_finite_loss_and_the_rows_they_trained_on(pipeline, words_dataset, slices):
    """A step event has to say enough to act on, or the caller cannot own the loop.

    Oracle: the dataset's own rows for each slice. The event names what the model read
    and what it was taught to write, as two JSON lists, and has no ``word`` key: a row is
    not one string.
    """
    events = list(pipeline.train_steps(words_dataset, slices, epoch=0))
    assert len(events) == SLICES
    for event, batch_slice in zip(events, slices, strict=True):
        assert math.isfinite(float(event.metrics["loss"]))
        rows = words_dataset.examples[batch_slice]
        assert json.loads(event.metrics["input"]) == [row.input for row in rows]
        assert json.loads(event.metrics["target"]) == [row.target for row in rows]
        assert "word" not in event.metrics


REFUSED = [
    (slice(0, 16, 2), "has step 2; a batch is contiguous rows"),
    (slice(7296, 7301), "ends past the dataset, which has 7300 rows"),
    (slice(8, 8), "is empty; there is nothing to step over"),
    (slice(12, 4), "is empty; there is nothing to step over"),
]


def test_a_slice_the_pipeline_cannot_step_over_is_refused_by_name(pipeline, words_dataset):
    """Oracle: the rule. The caller supplies the slices, so each kind the pipeline does
    not take raises from ``single_step`` and from ``evaluate`` alike, naming the slice,
    and no step is taken."""
    import torch

    before = [parameter.detach().clone() for parameter in pipeline.model.parameters()]
    for batch_slice, message in REFUSED:
        for run in (pipeline.single_step, pipeline.evaluate):
            with pytest.raises(ValueError) as raised:
                run(words_dataset, batch_slice)
            assert f"batch_slice {batch_slice} {message}" in str(raised.value), run.__name__
    after = pipeline.model.parameters()
    assert all(torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert len(json.loads(pipeline.evaluate(words_dataset, slice(7296, 7300))["input"])) == 4


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
