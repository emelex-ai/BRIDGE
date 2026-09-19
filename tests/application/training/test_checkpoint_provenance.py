"""Phoneme-table drift is detectable when a checkpoint is loaded.

Phoneme row ids and feature indices are positions in ``phonreps.csv``. Editing that file
relabels every one of them, and nothing else notices: no parameter shape changes, and the
derived feature matrix is a non-persistent buffer, so ``load_state_dict`` succeeds under
``strict=True``. The fingerprint recorded in ``VocabSpec`` is the only signal, and it is
only meaningful where two independently-produced tables meet: at the checkpoint boundary.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from tests.vocab import PHONEME_TABLE, TEST_VOCAB

LOGGER = "bridge.application.training.training_pipeline"


@pytest.fixture
def pipeline():
    """A bare object carrying only what the drift check touches."""
    stub = SimpleNamespace(
        model=Model(ModelConfig(vocab=TEST_VOCAB, d_model=16, nhead=2, seed=1)),
        logger=logging.getLogger(LOGGER),
    )
    stub._warn_on_phoneme_table_drift = TrainingPipeline._warn_on_phoneme_table_drift.__get__(stub)
    return stub


def checkpoint(fingerprint, missing=False):
    vocab = TEST_VOCAB.model_copy(update={"phon_table_fingerprint": fingerprint})
    if missing:  # pickle restores __dict__ verbatim, so the field can be absent entirely
        del vocab.__dict__["phon_table_fingerprint"]
    return {"model_config": SimpleNamespace(vocab=vocab)}


def test_a_stale_fingerprint_warns(pipeline, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline._warn_on_phoneme_table_drift(checkpoint("deadbeefdeadbeef"), "model.pth")
    assert "deadbeefdeadbeef" in caplog.text
    assert PHONEME_TABLE.fingerprint in caplog.text
    assert "model.pth" in caplog.text


def test_a_matching_fingerprint_is_silent(pipeline, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline._warn_on_phoneme_table_drift(checkpoint(PHONEME_TABLE.fingerprint), "model.pth")
    assert caplog.text == ""


@pytest.mark.parametrize("ckpt", [checkpoint(None), checkpoint(None, missing=True), {}])
def test_a_checkpoint_predating_the_field_is_silent(pipeline, caplog, ckpt):
    """Absent, None, and no config at all must all load without noise, and without
    raising, which a bare attribute read on an old pickled spec would do."""
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline._warn_on_phoneme_table_drift(ckpt, "model.pth")
    assert caplog.text == ""


def test_load_model_runs_the_drift_check_on_a_real_checkpoint(
    words_dataset, make_pipeline, tmp_path, caplog
):
    """The check is wired into the door a foreign checkpoint actually comes through.

    This replaces a test that read ``TrainingPipeline``'s source and asserted the string
    ``_warn_on_phoneme_table_drift`` appeared inside two method bodies. That had no oracle
    beyond "the text is in the file", and one of the two methods it guarded was never
    called by anything. Driving a real checkpoint through the real ``load_model`` is the
    claim the source search was standing in for.

    The control is the second half: a checkpoint whose fingerprint matches must load in
    silence, so the warning above is evidence of drift detection rather than of a loader
    that warns unconditionally.
    """
    pipeline = make_pipeline(words_dataset, model_artifacts_dir=str(tmp_path))
    bundle = {
        "model_config": pipeline.model.model_config,
        "dataset_config": words_dataset.dataset_config,
        "model_state_dict": pipeline.model.state_dict(),
        "optimizer_state_dict": pipeline.optimizer.state_dict(),
        "epoch": 0,
    }

    stale = bundle["model_config"].model_copy(deep=True)
    stale.vocab.phon_table_fingerprint = "0000000000000000"
    drifted = tmp_path / "drifted.pth"
    torch.save({**bundle, "model_config": stale}, drifted)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline.load_model(str(drifted))
    assert "Phoneme feature table mismatch" in caplog.text
    assert PHONEME_TABLE.fingerprint in caplog.text, "the warning names the current table"

    matching = tmp_path / "matching.pth"
    torch.save(bundle, matching)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        pipeline.load_model(str(matching))
    assert caplog.text == "", "a checkpoint from the same table must load in silence"
