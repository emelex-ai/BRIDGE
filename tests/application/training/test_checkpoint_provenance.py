"""Phoneme-table drift is detectable when a checkpoint is loaded.

Phoneme row ids and feature indices are positions in ``phonreps.csv``. Editing that file
relabels every one of them, and nothing else notices: no parameter shape changes, and the
derived feature matrix is a non-persistent buffer, so ``load_state_dict`` succeeds under
``strict=True``. The fingerprint recorded in ``VocabSpec`` is the only signal, and it is
only meaningful where two independently-produced tables meet: at the checkpoint boundary.
"""

import inspect
import logging
from types import SimpleNamespace

import pytest

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


def test_both_checkpoint_readers_run_the_drift_check():
    """``load_model`` is not the only door a foreign checkpoint comes through.

    ``transfer_partial_model_parameters`` copies selected modules out of someone else's
    checkpoint, which is the same silent-corruption case: shapes still match, so
    ``load_state_dict`` succeeds and nothing else would notice a relabelled feature table.
    """
    source = inspect.getsource(TrainingPipeline)
    for reader in ("load_model", "transfer_partial_model_parameters"):
        body = source.split(f"def {reader}(", 1)[1].split("\n    def ", 1)[0]
        assert "_warn_on_phoneme_table_drift" in body, f"{reader} skips the drift check"
