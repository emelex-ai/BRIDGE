"""
Behavioral tests pinning down the contract that `ModelConfig` (specifically
its `vocab` field) is the sole source of truth for the model's vocabulary
architecture: embedding dimensions, parameter counts, and special-token IDs
all flow from the config — not from any tokenizer or dataset reference.

These tests survived the decoupling refactor and provide permanent regression
coverage for the model↔tokenizer boundary. (The migration-only scaffolding
tests — AST scans, signature checks, attribute checks — were intentionally
deleted in Step 8 of `plans/sleepy-wishing-bird.md`: code review enforces
their invariants more reliably and they become brittle to legitimate
reorganization.)
"""

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model


def _build_test_model(**vocab_overrides) -> Model:
    """Construct a Model with controllable vocab numbers for testing."""
    defaults = {
        "orth_vocab_size": 109,
        "phon_vocab_size": 36,
        "orth_pad_id": 2,
        "orth_bos_id": 0,
        "orth_eos_id": 1,
        "orth_spc_id": 92,
        "phon_pad_id": 35,
        "phon_bos_id": 31,
        "phon_eos_id": 32,
        "phon_spc_id": 34,
    }
    defaults.update(vocab_overrides)
    return Model(ModelConfig(d_model=64, nhead=2, d_embedding=1, vocab=VocabSpec(**defaults)))


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
        model = _build_test_model(phon_eos_id=99, phon_pad_id=100, phon_bos_id=101)
        assert _find_in_pydantic(model.model_config, "phon_eos_id") == 99
        assert _find_in_pydantic(model.model_config, "phon_pad_id") == 100
        assert _find_in_pydantic(model.model_config, "phon_bos_id") == 101
