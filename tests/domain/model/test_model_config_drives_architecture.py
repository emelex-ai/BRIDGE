"""
Behavioral tests pinning down the contract that `ModelConfig` (specifically
its `vocab` field) is the model's source of truth for vocabulary architecture:
embedding dimensions, parameter counts, and special-token IDs all flow from the
config, not from any tokenizer or dataset reference. The phonological half must
agree with `phonreps.csv`, which the model checks at construction.

These tests survived the decoupling refactor and provide permanent regression
coverage for the model↔tokenizer boundary. (The migration-only scaffolding
tests (AST scans, signature checks, attribute checks) were intentionally
deleted in Step 8 of `plans/sleepy-wishing-bird.md`: code review enforces
their invariants more reliably and they become brittle to legitimate
reorganization.)
"""

import pytest

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from tests.vocab import PHONEME_TABLE, TEST_VOCAB


def _build_test_model(**vocab_overrides) -> Model:
    """Construct a Model with controllable vocab numbers for testing.

    The orthographic half is free; the phonological half is derived from
    ``phonreps.csv`` and is validated at construction, so it comes from TEST_VOCAB
    rather than being hand-written here.
    """
    vocab = TEST_VOCAB.model_copy(update={"orth_vocab_size": 109, **vocab_overrides})
    return Model(ModelConfig(d_model=64, nhead=2, d_embedding=1, vocab=vocab))


def _find_in_pydantic(model_obj, attr_name):
    """Recursively search a pydantic model for an attribute. Returns the value
    if found, else None. Lets tests assert that a value lives 'somewhere on
    the config' without prescribing flat vs nested shape."""
    if hasattr(model_obj, attr_name):
        candidate = getattr(model_obj, attr_name)
        if not hasattr(type(candidate), "model_fields"):
            return candidate
    if hasattr(type(model_obj), "model_fields"):
        for fname in type(model_obj).model_fields:
            sub = getattr(model_obj, fname, None)
            if hasattr(type(sub), "model_fields"):
                found = _find_in_pydantic(sub, attr_name)
                if found is not None:
                    return found
    return None


class TestConfigDrivesArchitecture:
    """Vocab sizes in the config control model construction."""

    def test_embedding_layer_sizes_match_config_vocab(self):
        model = _build_test_model(orth_vocab_size=50, phon_vocab_size=36)
        assert model.orthography_embedding.num_embeddings == 50
        assert model.phonology_embedding.num_embeddings == 36

    def test_changing_vocab_size_in_config_changes_param_count(self):
        small = _build_test_model(orth_vocab_size=50)
        large = _build_test_model(orth_vocab_size=500)
        n_small = sum(p.numel() for p in small.parameters())
        n_large = sum(p.numel() for p in large.parameters())
        assert n_large > n_small

    def test_model_config_carries_special_token_ids(self):
        """Sentinel special-token IDs supplied via config are reachable on
        `model.model_config` after construction (regardless of whether they
        live flat on the config or nested under a sub-object)."""
        model = _build_test_model()
        for field, token in (
            ("phon_eos_id", "[EOS]"),
            ("phon_pad_id", "[PAD]"),
            ("phon_bos_id", "[BOS]"),
        ):
            assert _find_in_pydantic(model.model_config, field) == PHONEME_TABLE.feature_of(token)

    def test_special_token_ids_must_match_the_phoneme_table(self):
        """Ids are not free: generation indexes the feature space with them directly, so a
        wrong-but-in-range id silently produces the wrong phoneme rather than an error."""
        with pytest.raises(ValueError, match="special-token ids disagree"):
            _build_test_model(phon_bos_id=PHONEME_TABLE.feature_of("[EOS]"))
