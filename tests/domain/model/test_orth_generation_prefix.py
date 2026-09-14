"""The orthographic decoder generates from the prefix it was trained on (issue #228).

``CharacterTokenizer`` lays every decoder sequence out as ``[LANG, BOS, ...chars, PAD...]``,
so during teacher forcing ``[BOS]`` sits at position 1 and the position-0 slot holds a
language token. Generation used to seed a lone ``[BOS]`` at position 0, which asked the
decoder to continue from a state it never saw in training and withheld the language token
entirely.

Phonology never had this problem, which is what made it visible: its decoder input is
``[BOS, ...phonemes, PAD...]`` with no language slot, so its ``[BOS]`` is at position 0 in
both regimes. Orthography was the odd one out, not both.

Two things are pinned here and they are different claims.

**Register.** The seed is the same two positions training uses, so the decoder starts where it
was taught to start. Measured on a model trained to memorise a closed 8-word list: the old
seeding reproduced 5 of 8 and derailed on two words entirely, `table` coming out `telephant`;
seeding the training prefix reproduces 8 of 8.

**Conditioning.** Keeping the language token on the decoder is what lets ``p2o`` spell the
same phonemes differently per language, which is the whole reason a language token exists.
The lexicons contain 58 phoneme sequences shared by English and Spanish with different
spellings. Trained on those and generated twice, once per language seed: 110 of 116 correct
with the prefix, against 64 of 116 for a decoder with no language input, which also cannot
fit the training data at all (loss 0.168 against 0.0065) because one input maps to two
answers it has no way to tell apart.

Full numbers in ``docs/decisions/0008-generation-seeds-the-training-prefix.md``.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.model import ORTH_DECODING
from bridge.domain.tokenizer import BridgeTokenizer

WORDS = ["long", "pencil", "hello world"]


@pytest.fixture(scope="module")
def tokenizer():
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def model(tokenizer):
    built = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(tokenizer), d_model=32, nhead=2, seed=5)
    )
    built.eval()
    return built


def names(tokenizer, ids):
    return [tokenizer.char_tokenizer.idx_2_char[int(i)] for i in ids]


@pytest.mark.parametrize("pathway", sorted(ORTH_DECODING))
def test_generation_opens_with_the_same_prefix_training_uses(tokenizer, model, pathway):
    """The claim, stated as the equality it is.

    Not "starts with a language token then BOS", which a hand-built seed would also satisfy
    while drifting from the tokenizer. The generated prefix must equal the tokenizer's own
    ``dec_input_ids[:, :2]`` for these words, so the two cannot diverge.
    """
    encoding = tokenizer.encode(WORDS)
    assert encoding is not None

    result = model.generate(encoding, pathway, deterministic=True)

    expected_prefix = encoding.orthographic.dec_input_ids[:, :2]
    assert torch.equal(result.orth_tokens[:, :2], expected_prefix), (
        f"{pathway} opened with {result.orth_tokens[:, :2].tolist()}, "
        f"training uses {expected_prefix.tolist()}"
    )


def test_the_prefix_is_the_language_the_caller_asked_for(tokenizer, model):
    """Conditioning, at the level this test can see it: the seed carries the caller's choice.

    Three languages over the same word, so a seed that ignored ``language_map`` and hardcoded
    one token would fail. Whether the trained model then *uses* the token is a question about
    training, measured in decision record 0008 rather than here, because it needs a trained
    model and a multilingual corpus.
    """
    seen = {}
    for language in ("--", "EN", "ES"):
        encoding = tokenizer.encode(["read"], language_map={"read": language})
        assert encoding is not None
        result = model.generate(encoding, "p2o", deterministic=True)
        seen[language] = names(tokenizer, result.orth_tokens[0, :2])

    assert seen == {
        "--": ["--", "[BOS]"],
        "EN": ["EN", "[BOS]"],
        "ES": ["ES", "[BOS]"],
    }, seen


def test_a_phonology_only_encoding_still_supplies_a_usable_prefix(tokenizer, model):
    """``p2o`` is the one pathway that emits orthography without consuming any.

    Its encoding carries a placeholder orthographic component, and a placeholder that could
    not supply a prefix would force a special case into the model for exactly this case. The
    placeholder is ``[--, BOS]``: unspecified language, which is the honest default when the
    caller never said.
    """
    encoding = tokenizer.encode(WORDS, modality_filter="phonology")
    assert encoding is not None

    result = model.generate(encoding, "p2o", deterministic=True)

    assert names(tokenizer, result.orth_tokens[0, :2]) == ["--", "[BOS]"]


@pytest.mark.parametrize("pathway", sorted(ORTH_DECODING))
def test_the_probability_history_stays_aligned_with_the_tokens(tokenizer, model, pathway):
    """One distribution per emitted token, including the seeded ones.

    The seed placeholders used to be a single hardcoded row certain of BOS. With a two-token
    prefix that would leave the histories off by one, which is invisible unless something
    counts them. Each seeded row must be certain of the token actually seeded there, so this
    also catches a placeholder built for the wrong token.
    """
    encoding = tokenizer.encode(WORDS)
    assert encoding is not None

    result = model.generate(encoding, pathway, deterministic=True)

    for item, (tokens, history) in enumerate(
        zip(result.orth_tokens, result.orth_probs, strict=True)
    ):
        assert len(history) <= tokens.shape[0]
        for position in range(2):
            row = history[position]
            assert torch.argmax(row).item() == tokens[position].item(), (
                f"{pathway} item {item}: seeded probability row {position} does not name "
                f"the token seeded there"
            )
            assert row.sum().item() == pytest.approx(1.0)


def test_generation_never_exceeds_the_position_table(tokenizer, model):
    """The step budget shrank by one when the seed grew by one, and must stay bounded.

    ``_positions`` raises when a sequence outruns the position table, so a loop that still
    budgeted ``max_orth_seq_len - 1`` steps on top of a two-token seed would raise here
    rather than silently truncate.
    """
    encoding = tokenizer.encode(WORDS)
    assert encoding is not None

    for pathway in sorted(ORTH_DECODING):
        result = model.generate(encoding, pathway, deterministic=True)
        assert result.orth_tokens.shape[1] <= model.max_orth_seq_len


def test_seeding_the_decoder_directly_without_a_prefix_is_refused(model, tokenizer):
    """``_generate`` is reachable on its own, and a missing prefix must say so.

    Silently falling back to a lone ``[BOS]`` would reintroduce the defect for any caller
    that bypasses ``generate``.
    """
    encoding = tokenizer.encode(WORDS)
    assert encoding is not None
    phonological = encoding.phonological

    with pytest.raises(ValueError, match="orth_dec_prefix"):
        model._generate(
            pathway="p2o",
            phon_enc_input=phonological.enc_input_ids,
            phon_enc_pad_mask=phonological.enc_pad_mask,
            deterministic=True,
        )
