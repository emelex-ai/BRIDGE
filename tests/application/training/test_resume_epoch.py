"""Resuming from a checkpoint has to actually resume.

``load_model`` reads the epoch out of the checkpoint, logs "Resuming training from epoch N",
and then used to throw the number away on the next line, so a run resumed from epoch 9 of 12
re-trained all twelve while the log said it started at ten. Nothing raised and nothing looked
wrong. The library no longer runs the loop, so ``start_epoch`` is now a value the caller
reads rather than one the library consumes, and it still has to be right.

The guard in front of the assignment needs every branch exercised, because a branch that
never fires and a branch that always fires look identical from the outside:

* a plain checkpoint resumes at ``epoch + 1``;
* a path naming a pretraining or finetuning checkpoint restarts at 0 on purpose, since those
  weights are being carried into a different run and the old epoch count means nothing there;
* a checkpoint carrying no epoch stays at 0 and says so in the log;
* a load that throws propagates the exception. A failed load and a deliberate reset used
  to be indistinguishable, because ``load_model`` swallowed everything and returned a
  ``False`` that ``__init__`` never checked.

Every numeric assertion here is paired with a control in the same test: the same recipe under
a condition that must come out the other way. The oracle throughout is the arithmetic stated
in the code's own comment, "Start from the next epoch", plus ``range(start, stop)``.
"""

import logging
import pickle

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model

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
    words = ["long", "pencil", "cat", "dog", "read", "write", "sun", "moon"]
    csv.write_text("word_raw,target_raw\n" + "".join(f"{word},{word}\n" for word in words))
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=str(csv)))


def build_pipeline(dataset, artifacts_dir, checkpoint_path=None, num_epochs=2):
    """A real ``TrainingPipeline``. ``__init__`` calls ``load_model`` when a checkpoint
    path is set, which is the code path a resumed run takes."""
    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
    model = Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5))
    training_config = TrainingConfig(
        training_pathway="o2p",
        model_artifacts_dir=str(artifacts_dir),
        checkpoint_path=checkpoint_path,
    )
    return TrainingPipeline(
        model=model,
        training_config=training_config,
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
    pipeline.load_model(path)


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

    transfer.load_model(transfer_path)
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
    pipeline.load_model(epochless)
    assert "epoch" in caplog.text.lower()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        control = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=with_epoch)
    assert control.start_epoch == NEXT_EPOCH
    assert caplog.text == ""


def test_a_corrupt_checkpoint_raises_rather_than_resuming_from_nothing(dataset, state, tmp_path):
    """A load that failed and a load that deliberately reset the counter used to leave
    identical state behind, because ``load_model`` swallowed every exception and returned
    ``False`` that nobody checked. A run then trained from random weights believing it had
    resumed. It raises now, so the two are distinguishable by construction.

    The control is a good checkpoint through the same pipeline, which must still reach
    epoch 10 rather than raising.
    """
    corrupt = tmp_path / "corrupt.pth"
    corrupt.write_bytes(b"not a torch checkpoint")
    good = write_checkpoint(state, tmp_path / "good" / "model_epoch_9.pth")

    pipeline = build_pipeline(dataset, tmp_path / "artifacts")

    # Pinned to the type, not to `Exception`: the point is that the failure reaches the
    # caller intact, and `torch.load` on a non-checkpoint raises UnpicklingError.
    with pytest.raises(pickle.UnpicklingError, match="invalid load key"):
        pipeline.load_model(str(corrupt))
    assert pipeline.start_epoch == 0

    pipeline.training_config.checkpoint_path = good
    pipeline.load_model(good)
    assert pipeline.start_epoch == NEXT_EPOCH


def test_a_checkpoint_of_another_model_is_refused_by_name_and_the_model_left_alone(
    dataset, state, tmp_path
):
    """Oracle: the rule that nothing loads with a tensor missing or left over, and the
    counts, worked out by hand.

    The checkpoint holds two tables of position embeddings, which this model has no place
    for, and lacks ``global_embedding``, which it has. That is 2 of the checkpoint's 168
    tensors and 1 of the model's 167. The error gives both counts and names the first of
    each, in a message short enough to read.

    The refusal comes before any weight is copied. ``linear_orthography_decoder.weight``
    is in both and differs in the checkpoint, and after the refusal the model still holds
    its own. Left to ``load_state_dict``, it would hold the checkpoint's.

    The control is the unedited checkpoint through the same pipeline, which loads.
    """
    shared = "linear_orthography_decoder.weight"
    weights = dict(state["model_state_dict"])
    weights["orth_position_embedding.weight"] = torch.zeros(30, 32)
    weights["phon_position_embedding.weight"] = torch.zeros(30, 32)
    weights[shared] = weights[shared] + 1.0
    del weights["global_embedding"]
    bad = write_checkpoint(
        {**state, "model_state_dict": weights}, tmp_path / "bad" / "model_epoch_9.pth"
    )
    good = write_checkpoint(state, tmp_path / "good" / "model_epoch_9.pth")

    pipeline = build_pipeline(dataset, tmp_path / "artifacts")
    before = pipeline.model.state_dict()[shared].clone()
    with pytest.raises(RuntimeError) as raised:
        pipeline.load_model(bad)

    message = str(raised.value)
    assert message == (
        f"{bad} does not hold this model's weights. Tensors in it with no place in the "
        "model: 2 of 168, the first 'orth_position_embedding.weight'. Tensors of the model "
        "missing from it: 1 of 167, the first 'global_embedding'."
    )
    assert torch.equal(pipeline.model.state_dict()[shared], before)
    assert pipeline.start_epoch == 0

    pipeline.load_model(good)
    assert pipeline.start_epoch == NEXT_EPOCH


@pytest.mark.parametrize(
    ("fault", "expected"),
    [
        ("left over", "Tensors in it with no place in the model: 1 of 168, the first 'extra'."),
        (
            "missing",
            "Tensors of the model missing from it: 1 of 167, the first 'global_embedding'.",
        ),
    ],
    ids=["left_over", "missing"],
)
def test_a_checkpoint_wrong_in_one_way_reports_that_way_only(
    dataset, state, tmp_path, fault, expected
):
    """Each half of the message alone: a tensor left over, and a tensor missing."""
    weights = dict(state["model_state_dict"])
    if fault == "left over":
        weights["extra"] = torch.zeros(1)
    else:
        del weights["global_embedding"]
    bad = write_checkpoint({**state, "model_state_dict": weights}, tmp_path / "model_epoch_9.pth")

    with pytest.raises(RuntimeError) as raised:
        build_pipeline(dataset, tmp_path / "artifacts").load_model(bad)
    assert str(raised.value) == f"{bad} does not hold this model's weights. {expected}"


def test_a_checkpoint_of_another_shape_is_refused_and_no_tensor_is_overwritten(
    dataset, state, tmp_path
):
    """Oracle: the rule that a refused load leaves the model as it was, checked on every
    tensor, bitwise.

    The checkpoint names exactly the model's tensors, and one has another shape: its
    ``global_embedding`` holds one memory row where the model holds two, which is what a
    checkpoint saved at ``d_embedding=1`` looks like to a model built at the default.
    Every other tensor in it fits and differs from the model's by one.

    The control is ``load_state_dict`` on a model built the same way. It raises as well,
    after copying in the tensors that fit, so an overwrite is something this test sees.
    """
    weights = {name: tensor + 1.0 for name, tensor in state["model_state_dict"].items()}
    weights["global_embedding"] = weights["global_embedding"][:, :1]
    bad = write_checkpoint({**state, "model_state_dict": weights}, tmp_path / "model_epoch_9.pth")

    def overwritten(model, before) -> int:
        after = model.state_dict()
        return sum(not torch.equal(before[name], after[name]) for name in before)

    pipeline = build_pipeline(dataset, tmp_path / "artifacts")
    before = {name: tensor.clone() for name, tensor in pipeline.model.state_dict().items()}
    with pytest.raises(RuntimeError) as raised:
        pipeline.load_model(bad)

    assert str(raised.value) == (
        f"{bad} does not hold this model's weights. Tensors of another shape: 1 of 167, "
        "the first 'global_embedding', (1, 1, 32) where the model has (1, 2, 32)."
    )
    assert overwritten(pipeline.model, before) == 0
    assert pipeline.start_epoch == 0

    control = build_pipeline(dataset, tmp_path / "artifacts").model
    with pytest.raises(RuntimeError, match="size mismatch for global_embedding"):
        control.load_state_dict(weights)
    assert overwritten(control, before) > 0, "control: torch's own refusal overwrites nothing"


def test_start_epoch_is_what_a_callers_own_loop_resumes_from(dataset, state, tmp_path):
    """The consequence of the field, now that the loop belongs to the caller.

    The library used to iterate ``range(self.start_epoch, num_epochs)`` itself. It does
    not run a loop any more (docs/decisions/0013), so what has to hold is that
    ``start_epoch`` is correct and reachable, and that a caller writing the obvious
    ``range(pipeline.start_epoch, n)`` gets the remaining epochs and no others.

    Oracle: ``range`` itself, with both expected lists written out in full. The control is
    the second pipeline, built identically but with no checkpoint, which must run all
    twelve.
    """
    path = write_checkpoint(state, tmp_path / "model_epoch_9.pth")
    resumed = build_pipeline(dataset, tmp_path / "artifacts", checkpoint_path=path)
    fresh = build_pipeline(dataset, tmp_path / "artifacts")

    assert list(range(resumed.start_epoch, 12)) == [10, 11]
    assert list(range(fresh.start_epoch, 12)) == list(range(12))
