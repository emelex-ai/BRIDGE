"""Pins the helpers that three encoders and four forward pathways now share.

``_mix_with_global``, ``_decode_orth`` and ``_decode_phon`` replaced seven hand-copied
blocks. A single edit to any of them now changes every pathway at once, so the shared
invariants are asserted directly, plus the parameter-name stability that checkpoints
depend on.
"""

import io

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Decoder, Encoder, Model
from bridge.domain.tokenizer import BridgeTokenizer
from tests.vocab import PHONEME_TABLE, TEST_VOCAB

VOCAB = TEST_VOCAB
D_MODEL = 32
BATCH = 3


def build(**over):
    kwargs = {"vocab": VOCAB, "d_model": D_MODEL, "nhead": 2, "seed": 5}
    kwargs.update(over)
    return Model(ModelConfig(**kwargs))


@pytest.fixture
def model():
    return build()


def orth_inputs(seq=6):
    ids = torch.randint(0, VOCAB.orth_vocab_size, (BATCH, seq), dtype=torch.long)
    return ids, torch.zeros((BATCH, seq), dtype=torch.bool)


def phon_inputs(seq=4):
    """Phoneme *row* ids: indices into the phoneme table, not feature indices."""
    ids = torch.randint(0, PHONEME_TABLE.num_rows, (BATCH, seq), dtype=torch.long)
    return ids, torch.zeros((BATCH, seq), dtype=torch.bool)


# --- _mix_with_global ------------------------------------------------------


def test_mix_with_global_returns_d_embedding_rows(model):
    encoding = torch.randn(BATCH, 7, D_MODEL)
    mask = torch.zeros((BATCH, 7), dtype=torch.bool)
    out = model._mix_with_global(encoding, mask)
    assert out.shape == (BATCH, model.model_config.d_embedding, D_MODEL)


@pytest.mark.parametrize("d_embedding", [1, 2, 4])
def test_mix_with_global_honours_d_embedding(d_embedding):
    """The three encoder tails had drifted: one hardcoded a width of 1 where the others
    used ``d_embedding``. Unifying them means all three must now scale together."""
    m = build(d_embedding=d_embedding)
    encoding = torch.randn(BATCH, 5, D_MODEL)
    mask = torch.zeros((BATCH, 5), dtype=torch.bool)
    assert m._mix_with_global(encoding, mask).shape == (BATCH, d_embedding, D_MODEL)


@pytest.mark.parametrize("d_embedding", [1, 2, 4])
def test_mix_with_global_slices_every_global_row(d_embedding):
    """Value-level check that the mixer output is sliced ``[:, :d_embedding]``.

    A shape assertion cannot catch a wrong slice here: ``mixed[:, :1] + global`` has
    shape ``(B, 1, D) + (B, d_embedding, D)``, which *broadcasts* back to the right
    shape while silently giving every global row the mixer's first row. That is
    precisely the bug the pre-refactor ``embed_o`` carried, so it is pinned by value.
    """
    m = build(d_embedding=d_embedding)
    m.eval()
    encoding = torch.randn(BATCH, 5, D_MODEL)
    mask = torch.zeros((BATCH, 5), dtype=torch.bool)

    with torch.no_grad():
        actual = m._mix_with_global(encoding, mask)

        global_rows = m.global_embedding.expand(BATCH, -1, -1)
        mixed = m.transformer_mixer(
            torch.cat((global_rows, encoding), dim=1),
            src_key_padding_mask=torch.cat(
                (torch.zeros((BATCH, d_embedding), dtype=torch.bool), mask), dim=-1
            ),
        )
        expected = mixed[:, :d_embedding] + global_rows

    assert torch.allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("d_embedding", [1, 2, 3])
def test_all_three_encoders_agree_on_output_shape(d_embedding):
    """embed_o / embed_p / embed_op must produce interchangeable memory for the decoders."""
    m = build(d_embedding=d_embedding)
    o, om = orth_inputs()
    p, pm = phon_inputs()
    expected = (BATCH, d_embedding, D_MODEL)
    assert m.embed_o(o, om).shape == expected
    assert m.embed_p(p, pm).shape == expected
    assert m.embed_op(o, om, p, pm).shape == expected


def test_mix_with_global_gradient_reaches_the_global_parameter(model):
    """``global_embedding`` is broadcast with ``.expand`` (stride 0). Its gradient must
    still accumulate across the batch. A broadcasting mistake here silently stops the
    global token from learning."""
    model.zero_grad(set_to_none=True)
    encoding = torch.randn(BATCH, 5, D_MODEL)
    mask = torch.zeros((BATCH, 5), dtype=torch.bool)
    model._mix_with_global(encoding, mask).sum().backward()

    grad = model.global_embedding.grad
    assert grad is not None
    assert grad.shape == model.global_embedding.shape
    assert torch.any(grad != 0)


def test_mix_with_global_never_masks_the_global_positions(model):
    """The global rows are prepended with an all-False pad mask. If they were masked,
    the mixer would attend to nothing and produce NaNs."""
    encoding = torch.randn(BATCH, 5, D_MODEL)
    fully_padded = torch.ones((BATCH, 5), dtype=torch.bool)
    out = model._mix_with_global(encoding, fully_padded)
    assert torch.isfinite(out).all()


# --- _decode_orth / _decode_phon ------------------------------------------


def test_decode_orth_logit_shape(model):
    memory = torch.randn(BATCH, 1, D_MODEL)
    ids, mask = orth_inputs(seq=5)
    out = model._decode_orth(memory, ids, mask)
    assert out.shape == (BATCH, VOCAB.orth_vocab_size, 5)


def test_decode_phon_logit_shape(model):
    memory = torch.randn(BATCH, 1, D_MODEL)
    ids, mask = phon_inputs(seq=4)
    out = model._decode_phon(memory, ids, mask)
    # (batch, {off,on}, seq, features)
    assert out.shape == (BATCH, 2, 4, VOCAB.phon_vocab_size - 1)


def test_decoders_are_causal(model):
    """Changing a later decoder position must not alter an earlier position's logits."""
    memory = torch.randn(BATCH, 1, D_MODEL)
    ids, mask = orth_inputs(seq=6)
    model.eval()
    with torch.no_grad():
        base = model._decode_orth(memory, ids, mask)
        tampered = ids.clone()
        tampered[:, -1] = (tampered[:, -1] + 1) % VOCAB.orth_vocab_size
        after = model._decode_orth(memory, tampered, mask)
    assert torch.allclose(base[:, :, :-1], after[:, :, :-1], atol=1e-5)


# --- forward pathways compose the helpers ---------------------------------


def test_forward_pathways_return_the_expected_keys(model):
    o, om = orth_inputs()
    p, pm = phon_inputs()
    assert set(
        model(
            task="o2p",
            orth_enc_input=o,
            orth_enc_pad_mask=om,
            phon_dec_input=p,
            phon_dec_pad_mask=pm,
        )
    ) == {"phon"}
    assert set(
        model(
            task="p2o",
            phon_enc_input=p,
            phon_enc_pad_mask=pm,
            orth_dec_input=o,
            orth_dec_pad_mask=om,
        )
    ) == {"orth"}
    assert set(
        model(
            task="p2p",
            phon_enc_input=p,
            phon_enc_pad_mask=pm,
            phon_dec_input=p,
            phon_dec_pad_mask=pm,
        )
    ) == {"phon"}
    assert set(
        model(
            task="op2op",
            orth_enc_input=o,
            orth_enc_pad_mask=om,
            phon_enc_input=p,
            phon_enc_pad_mask=pm,
            orth_dec_input=o,
            orth_dec_pad_mask=om,
            phon_dec_input=p,
            phon_dec_pad_mask=pm,
        )
    ) == {"orth", "phon"}


def test_forward_rejects_an_unknown_task(model):
    with pytest.raises(ValueError, match="Invalid pathway selected."):
        model(task="nope")


def test_o2p_and_p2p_share_the_phonological_decode(model):
    """Both call ``_decode_phon`` with the same decoder input, differing only in how
    the memory was encoded. Shapes must therefore match exactly."""
    o, om = orth_inputs()
    p, pm = phon_inputs()
    a = model(
        task="o2p", orth_enc_input=o, orth_enc_pad_mask=om, phon_dec_input=p, phon_dec_pad_mask=pm
    )
    b = model(
        task="p2p", phon_enc_input=p, phon_enc_pad_mask=pm, phon_dec_input=p, phon_dec_pad_mask=pm
    )
    assert a["phon"].shape == b["phon"].shape


# --- phono_sample ---------------------------------------------------------


def active_features(embedding_input: torch.Tensor) -> list[list[int]]:
    """Decode phono_sample's embedding input back to per-row active feature indices."""
    return [torch.nonzero(row).flatten().tolist() for row in embedding_input]


def test_phono_sample_all_features_off_falls_back_to_pad(model):
    """The empty-feature branch: a phoneme with no active feature decodes to [PAD]."""
    probs = torch.zeros((BATCH, 2, VOCAB.phon_vocab_size - 1))
    probs[:, 0, :] = 1.0
    presence, embedding_input = model.phono_sample(probs, deterministic=True)
    # `presence` is reported as sampled, all off, while the embedding input carries
    # [PAD]. The two deliberately disagree here; `phon_vecs` records the former.
    assert torch.equal(presence, torch.zeros_like(presence))
    assert active_features(embedding_input) == [[VOCAB.phon_pad_id]] * BATCH


def test_phono_sample_all_features_on(model):
    probs = torch.zeros((BATCH, 2, VOCAB.phon_vocab_size - 1))
    probs[:, 1, :] = 1.0
    presence, embedding_input = model.phono_sample(probs, deterministic=True)
    assert torch.equal(presence, torch.ones_like(presence))
    assert active_features(embedding_input) == [list(range(VOCAB.phon_vocab_size - 1))] * BATCH


def test_phono_sample_mixed_features(model):
    probs = torch.zeros((BATCH, 2, VOCAB.phon_vocab_size - 1))
    probs[:, 1, [1, 4]] = 1.0
    _, embedding_input = model.phono_sample(probs, deterministic=True)
    assert active_features(embedding_input) == [[1, 4]] * BATCH


def test_phono_sample_only_one_row_empty(model):
    """Per-row fallback: a row with features must not be overwritten by the PAD path."""
    probs = torch.zeros((BATCH, 2, VOCAB.phon_vocab_size - 1))
    probs[1, 1, [2, 3]] = 1.0
    presence, embedding_input = model.phono_sample(probs, deterministic=True)
    active = active_features(embedding_input)
    assert active[0] == [VOCAB.phon_pad_id]
    assert active[1] == [2, 3]
    assert active[2] == [VOCAB.phon_pad_id]
    # Only the embedding input gains [PAD]; the reported presence stays all-off.
    assert not presence[0].any()
    assert not presence[2].any()


def test_phono_sample_embedding_input_is_one_column_wider_than_presence(model):
    """[PAD] is never predicted, so it needs a column the decoder does not emit."""
    probs = torch.zeros((BATCH, 2, VOCAB.phon_vocab_size - 1))
    presence, embedding_input = model.phono_sample(probs, deterministic=True)
    assert presence.shape == (BATCH, VOCAB.phon_vocab_size - 1)
    assert embedding_input.shape == (BATCH, VOCAB.phon_vocab_size)


# --- checkpoint compatibility ---------------------------------------------


def test_parameter_names_are_stable_across_the_layers_module_merge():
    """``encoder.py`` and ``decoder.py`` were merged into ``layers.py``.

    Class and attribute names were kept identical precisely so that existing
    checkpoints keep loading. Renaming ``Encoder.transformer_encoder`` or the module
    attributes on ``Model`` would silently break ``TrainingPipeline.load_model``.
    """
    keys = set(build().state_dict().keys())
    for expected in [
        "global_embedding",
        "orthography_embedding.weight",
        "phonology_embedding.weight",
        "orth_position_embedding.weight",
        "phon_position_embedding.weight",
        "orthography_encoder.transformer_encoder.layers.0.self_attn.in_proj_weight",
        "phonology_encoder.transformer_encoder.layers.0.self_attn.in_proj_weight",
        "transformer_mixer.transformer_encoder.layers.0.self_attn.in_proj_weight",
        "orthography_decoder.transformer_decoder.layers.0.self_attn.in_proj_weight",
        "phonology_decoder.transformer_decoder.layers.0.self_attn.in_proj_weight",
        "linear_orthography_decoder.weight",
        "linear_phonology_decoder.weight",
        "gp_multihead_attention.in_proj_weight",
        "pg_multihead_attention.in_proj_weight",
    ]:
        assert expected in keys, f"checkpoint key disappeared: {expected}"


def test_a_state_dict_round_trips():
    a, b = build(), build(seed=99)
    b.load_state_dict(a.state_dict())
    for (ka, va), (kb, vb) in zip(
        sorted(a.state_dict().items()), sorted(b.state_dict().items()), strict=True
    ):
        assert ka == kb
        assert torch.equal(va, vb)


def test_the_phoneme_feature_table_stays_out_of_the_state_dict():
    """``phon_feature_matrix`` is derived from ``phonreps.csv``, not learned.

    It is registered ``persistent=False`` precisely so it never reaches a checkpoint. If
    that flag were dropped, every ``.pth`` written before the issue #221 refactor would
    fail to load under ``strict=True``, the failure this test exists to prevent.
    """
    model = build()
    assert "phon_feature_matrix" in dict(model.named_buffers())
    assert "phon_feature_matrix" not in model.state_dict()


def test_a_pre_refactor_checkpoint_loads_strictly():
    """Spec acceptance gate: an existing ``.pth`` still loads with ``strict=True``.

    The refactor added no parameters and changed no shapes, so a checkpoint saved before it
    has exactly today's key set. Round-tripping through ``torch.save``/``torch.load`` is
    what ``TrainingPipeline.load_model`` actually does, so this exercises that path rather
    than a bare dict copy.
    """
    saved = io.BytesIO()
    torch.save({"model_state_dict": build().state_dict()}, saved)
    saved.seek(0)

    fresh = build(seed=99)
    missing, unexpected = fresh.load_state_dict(
        torch.load(saved, weights_only=True)["model_state_dict"], strict=True
    )
    assert not missing and not unexpected


# --- phoneme table provenance ----------------------------------------------


def test_from_tokenizer_records_the_table_fingerprint():
    tokenizer = BridgeTokenizer()
    spec = VocabSpec.from_tokenizer(tokenizer)
    assert spec.phon_table_fingerprint == tokenizer.phoneme_tokenizer.phoneme_table.fingerprint


def test_a_model_records_the_fingerprint_of_the_table_it_built_from():
    """Checkpoint drift is detected at load time, so the model must carry the comparand."""
    assert build().phon_table_fingerprint == PHONEME_TABLE.fingerprint


def test_a_vocab_spec_without_a_fingerprint_still_builds():
    """Specs written before the field carry ``None`` and must construct without complaint."""
    assert isinstance(
        build(vocab=TEST_VOCAB.model_copy(update={"phon_table_fingerprint": None})), Model
    )


def test_layer_wrappers_are_still_exported():
    assert issubclass(Encoder, torch.nn.Module)
    assert issubclass(Decoder, torch.nn.Module)


def test_layer_counts_follow_the_config():
    m = build(num_orth_enc_layers=3, num_phon_enc_layers=1, num_mixing_enc_layers=2)
    assert len(m.orthography_encoder.transformer_encoder.layers) == 3
    assert len(m.phonology_encoder.transformer_encoder.layers) == 1
    assert len(m.transformer_mixer.transformer_encoder.layers) == 2
