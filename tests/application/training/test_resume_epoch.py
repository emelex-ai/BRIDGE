"""Resuming from a checkpoint has to actually resume.

``load_model`` reads the epoch out of the checkpoint, logs "Resuming training from epoch N",
and then throws the number away on the next line. ``run_train_val_loop`` iterates
``range(self.start_epoch, num_epochs)``, so a run resumed from epoch 9 of 12 re-trains all
twelve epochs while the log says it started at ten. Nothing raises and nothing looks wrong.

The guard in front of the assignment needs every branch exercised, because a branch that
never fires and a branch that always fires look identical from the outside:

* a plain checkpoint resumes at ``epoch + 1``;
* a path naming a pretraining or finetuning checkpoint restarts at 0 on purpose, since those
  weights are being carried into a different run and the old epoch count means nothing there;
* a checkpoint carrying no epoch stays at 0 and says so in the log;
* a load that throws stays at 0 and reports failure by returning False. The return value is
  the only difference between a failed load and a deliberate reset, because ``load_model``
  swallows the exception.

Every numeric assertion here is paired with a control in the same test: the same recipe under
a condition that must come out the other way. The oracle throughout is the arithmetic stated
in the code's own comment, "Start from the next epoch", plus ``range(start, stop)``.
"""

import logging

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.infra.metrics.metrics_logger import STDOutMetricsLogger

LOGGER = "bridge.application.training.training_pipeline"

# The epoch written into every checkpoint below, and the epoch a resumed run must start on.
# 9 is the last epoch finished, so the next one to run is 10.
SAVED_EPOCH = 9
NEXT_EPOCH = SAVED_EPOCH + 1


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """Eight common words. The dataset is only here so a real pipeline can be built;
    nothing in this file trains, and constructing the tokenizer is the slow part."""
    csv = tmp_path_factory.mktemp("data") / "words.csv"
    csv.write_text(
        "word_raw\n"
        + "\n".join(["long", "pencil", "cat", "dog", "read", "write", "sun", "moon"])
        + "\n"
    )
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=str(csv)))


def build_pipeline(dataset, artifacts_dir, checkpoint_path=None, num_epochs=2):
    """A real ``TrainingPipeline``. ``__init__`` calls ``load_model`` when a checkpoint
    path is set, which is the code path a resumed run takes."""
    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
    model = Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5))
    training_config = TrainingConfig(
        num_epochs=num_epochs,
        training_pathway="o2p",
        model_artifacts_dir=str(artifacts_dir),
        checkpoint_path=checkpoint_path,
    )
    metrics_config = MetricsConfig(
        batch_metrics=False,
        training_metrics=False,
        validation_metrics=False,
        modes=[],
        filename=None,
    )
    return TrainingPipeline(
        model=model,
        dataset=dataset,
        training_config=training_config,
        metrics_logger=STDOutMetricsLogger(metrics_config),
    )


@pytest.fixture(scope="module")
def state(dataset, tmp_path_factory):
    """Weights and optimizer state a checkpoint can carry. Taken from a pipeline built the
    same way as the ones that load them, so ``load_state_dict`` has nothing to complain
    about and any failure in these tests is about the epoch, not about the tensors."""
    pipeline = build_pipeline(dataset, tmp_path_factory.mktemp("artifacts"))
    return {
        "model_state_dict": pipeline.model.state_dict(),
        "optimizer_state_dict": pipeline.optimizer.state_dict(),
    }


def write_checkpoint(state, path, epoch=SAVED_EPOCH):
    """Write a checkpoint of the shape ``save_checkpoint`` writes. ``epoch=None`` omits the key
    entirely, which is what a checkpoint written before that field existed looks like."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    if epoch is not None:
        payload["epoch"] = epoch
    torch.save(payload, path)
    return str(path)


def test_a_plain_checkpoint_resumes_at_the_next_epoch(dataset, state, tmp_path):
    """Epoch 9 is finished, so the next epoch to run is 10.

    The checkpoint on disk really does record 9, asserted here so a failure below cannot be
    blamed on the fixture writing the wrong number.
    """
    path = write_checkpoint(state, tmp_path / "model_epoch_9.pth")
    assert torch.load(path, weights_only=False)["epoch"] == SAVED_EPOCH

    pipeline = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=path)

    assert pipeline.start_epoch == NEXT_EPOCH
    assert pipeline.load_model(path) is True


@pytest.mark.parametrize("kind", ["pretraining", "finetuning"])
def test_transfer_learning_checkpoints_restart_the_counter(dataset, state, tmp_path, kind):
    """A pretraining or finetuning checkpoint is weights being carried into a new run, so
    its epoch number belongs to a different training curve and must not be resumed.

    The control is the second half of this test: byte-identical checkpoint contents under a
    plain path must resume at 10. Without it, "start_epoch is 0" proves nothing, since a
    guard that is always false and a guard that is never false both produce 0 here.
    """
    transfer_path = write_checkpoint(state, tmp_path / kind / "model_epoch_9.pth")
    plain_path = write_checkpoint(state, tmp_path / "plain" / "model_epoch_9.pth")

    transfer = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=transfer_path)
    plain = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=plain_path)

    assert transfer.load_model(transfer_path) is True
    assert transfer.start_epoch == 0
    assert plain.start_epoch == NEXT_EPOCH
    assert transfer.start_epoch != plain.start_epoch


def test_a_checkpoint_without_an_epoch_key_starts_at_zero_and_warns(
    dataset, state, tmp_path, caplog
):
    """Old checkpoints carry no epoch. There is nothing to resume from, so 0 is right, but
    it has to be announced: silently restarting a long run is the expensive failure.

    The control is the same checkpoint written with the key present, which must give 10 and
    must not warn.
    """
    epochless = write_checkpoint(state, tmp_path / "epochless" / "model.pth", epoch=None)
    with_epoch = write_checkpoint(state, tmp_path / "with_epoch" / "model.pth")

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=epochless)
    assert pipeline.start_epoch == 0
    assert pipeline.load_model(epochless) is True
    assert "epoch" in caplog.text.lower()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        control = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=with_epoch)
    assert control.start_epoch == NEXT_EPOCH
    assert caplog.text == ""


def test_a_corrupt_checkpoint_reports_failure(dataset, state, tmp_path):
    """``load_model`` catches every exception, so a load that failed and a load that
    deliberately reset the counter leave identical state behind. The return value is the
    only thing that separates them, so it is what this asserts.

    The control is a good checkpoint through the same pipeline: True, and epoch 10.
    """
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a torch checkpoint")
    good = write_checkpoint(state, tmp_path / "good" / "model_epoch_9.pth")

    pipeline = build_pipeline(dataset, tmp_path / "artifacts")

    assert pipeline.load_model(str(corrupt)) is False
    assert pipeline.start_epoch == 0

    pipeline.training_config.checkpoint_path = good
    assert pipeline.load_model(good) is True
    assert pipeline.start_epoch == NEXT_EPOCH


def test_the_resumed_loop_runs_only_the_remaining_epochs(dataset, state, tmp_path):
    """The consequence of the field, not the field itself.

    ``run_train_val_loop`` iterates ``range(self.start_epoch, num_epochs)``. Resuming a
    12-epoch run from a checkpoint at epoch 9 must run epochs 10 and 11 and nothing else;
    a run with no checkpoint must run all twelve. The control is that second list, and the
    oracle is ``range`` itself, so both expected lists are written out in full.

    No training happens: ``train_steps`` is replaced by a recorder
    on the instance, because the claim is about the loop bounds and the bounds come from the
    real ``load_model``.
    """

    def epochs_run(pipeline):
        seen: list[int] = []
        pipeline.train_steps = lambda epoch: iter((seen.append(epoch), ())[1])
        pipeline.val_slices = []
        pipeline.test_dataset = None
        list(pipeline.run_train_val_loop())
        return seen

    path = write_checkpoint(state, tmp_path / "model_epoch_9.pth")
    resumed = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=path, num_epochs=12)
    fresh = build_pipeline(dataset, tmp_path / "artifacts", num_epochs=12)

    assert epochs_run(resumed) == [10, 11]
    assert epochs_run(fresh) == list(range(12))
