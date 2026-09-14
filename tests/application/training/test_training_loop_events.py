"""What ``run_train_val_loop`` emits, now that the caller owns the loop.

The generator used to yield one aggregated dict per epoch and checkpoint on its own
schedule. It now yields a :class:`TrainingEvent` per optimizer step and per boundary, and
writes nothing. That makes the stream the public contract, so these tests pin its shape:
how many events arrive, in what order, and what each carries.

See docs/decisions/0006-the-caller-owns-the-training-loop.md.
"""

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    TrainingEvent,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.infra.metrics.metrics_logger import STDOutMetricsLogger

DATA_CSV = "tests/domain/model/data/data.csv"
TRAIN_SLICES = 3
VAL_SLICES = 2

SILENT_METRICS = MetricsConfig(
    batch_metrics=False,
    training_metrics=False,
    validation_metrics=False,
    modes=[],
    filename=None,
)


@pytest.fixture(scope="module")
def dataset():
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA_CSV))


@pytest.fixture
def pipeline(dataset, tmp_path):
    """A pipeline cut down to a fixed, small number of slices.

    The counts below are asserted against these two constants rather than against
    ``len(pipeline.train_slices)`` read back at assertion time, so a loop that silently
    skipped slices could not satisfy them by agreeing with itself.
    """
    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
    built = TrainingPipeline(
        model=Model(ModelConfig(vocab=vocab, d_model=16, nhead=2, seed=5)),
        dataset=dataset,
        training_config=TrainingConfig(
            num_epochs=2,
            training_pathway="o2p",
            model_artifacts_dir=str(tmp_path),
            shuffle_each_epoch=False,
        ),
        metrics_logger=STDOutMetricsLogger(SILENT_METRICS),
    )
    built.train_slices = built.train_slices[:TRAIN_SLICES]
    built.val_slices = built.val_slices[:VAL_SLICES]
    return built


def test_the_stream_has_one_train_event_per_step_and_one_epoch_event_per_epoch(pipeline):
    """The arithmetic of the stream, counted against constants set in the fixture."""
    epochs = 2
    events = list(pipeline.run_train_val_loop(num_epochs=epochs))

    assert all(isinstance(event, TrainingEvent) for event in events)
    phases = [event.phase for event in events]
    assert phases.count("train") == TRAIN_SLICES * epochs
    assert phases.count("validation") == epochs
    assert phases.count("epoch") == epochs
    assert phases.count("test") == 0, "no test dataset is configured"
    assert len(events) == (TRAIN_SLICES + 2) * epochs


def test_each_epoch_ends_with_its_epoch_event(pipeline):
    """Ordering is part of the contract: a caller checkpointing on ``epoch`` needs the
    training steps for that epoch to have already happened when it arrives."""
    events = list(pipeline.run_train_val_loop(num_epochs=2))

    for epoch in (0, 1):
        of_epoch = [event for event in events if event.epoch == epoch]
        assert [event.phase for event in of_epoch] == (
            ["train"] * TRAIN_SLICES + ["validation", "epoch"]
        )
        assert [event.step for event in of_epoch[:TRAIN_SLICES]] == list(range(TRAIN_SLICES))

    assert [event.epoch for event in events] == sorted(event.epoch for event in events), (
        "epochs must arrive in order"
    )


def test_train_events_carry_a_finite_loss_and_the_words_they_trained_on(pipeline):
    """A step event has to say enough to act on, or the caller cannot own the loop."""
    events = [e for e in pipeline.run_train_val_loop(num_epochs=1) if e.phase == "train"]

    for event in events:
        assert "loss" in event.metrics
        assert torch.isfinite(event.metrics["loss"].detach()).all()
        assert "word" in event.metrics, "the batch's words are how a caller identifies a step"


def test_the_epoch_event_carries_the_merged_aggregate(pipeline):
    """The epoch row is what this generator yielded before it yielded per step, so a
    caller that only wants epoch rows filters on the phase and is otherwise unchanged."""
    events = list(pipeline.run_train_val_loop(num_epochs=1))
    epoch_event = next(e for e in events if e.phase == "epoch")

    assert any(key.startswith("train_") for key in epoch_event.metrics)
    assert any(key.startswith("valid_") for key in epoch_event.metrics)
    assert "train_time_per_step" in epoch_event.metrics

    # The mean of the step losses, computed here from the stream rather than read back.
    step_losses = [float(e.metrics["loss"].detach()) for e in events if e.phase == "train"]
    assert float(epoch_event.metrics["train_loss"].detach()) == pytest.approx(
        sum(step_losses) / len(step_losses), rel=1e-5
    )


def test_num_epochs_overrides_the_config(pipeline):
    """The convenience argument, since the caller owning the loop still wants the short
    form. The config says 2; the call says 1, and the call wins."""
    assert pipeline.training_config.num_epochs == 2

    events = list(pipeline.run_train_val_loop(num_epochs=1))

    assert {event.epoch for event in events} == {0}


def test_a_resumed_run_does_only_the_remaining_epochs(pipeline):
    """Counting starts at ``start_epoch``, so resume and the override compose."""
    pipeline.start_epoch = 3

    events = list(pipeline.run_train_val_loop(num_epochs=5))

    assert sorted({event.epoch for event in events}) == [3, 4]


def test_the_caller_can_stop_early_without_finishing_the_epoch(pipeline):
    """The point of a generator: abandoning it must not run the rest of the work.

    Measured by counting the optimizer steps that actually happened, not by trusting that
    breaking out of a for loop did what it looks like it does.
    """
    steps = 0
    original = pipeline.single_step

    def counting(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original(*args, **kwargs)

    pipeline.single_step = counting

    for event in pipeline.run_train_val_loop(num_epochs=2):
        if event.phase == "train" and event.step == 1:
            break

    assert steps == 2, f"expected to stop after 2 steps, ran {steps}"


def test_train_steps_runs_one_epoch_on_its_own(pipeline):
    """The seam underneath the convenience loop, usable directly.

    It does not shuffle: reordering belongs to the epoch, and a caller driving this across
    several epochs should not get a permutation it did not ask for.
    """
    before = list(pipeline.dataset.words)

    events = list(pipeline.train_steps(epoch=7))

    assert [e.phase for e in events] == ["train"] * TRAIN_SLICES
    assert [e.epoch for e in events] == [7] * TRAIN_SLICES
    assert [e.step for e in events] == list(range(TRAIN_SLICES))
    assert pipeline.dataset.words == before, "train_steps must not reorder the dataset"
