"""The orthographic decoder generates from the prefix it was trained on (issue #228).

``CharacterTokenizer`` lays every target out as ``[LANG, BOS, ...chars, PAD...]``, so
during teacher forcing ``[BOS]`` sits at position 1 and the position-0 slot holds the
language token of the row's first word. Generation is seeded with those same two
positions, which the caller builds with ``BridgeTokenizer.orth_start`` and passes as
``orth_start``. A lone ``[BOS]`` at position 0 would ask the decoder to continue from a
state it never saw in training and would withhold the language token entirely.

Phonology has no such gap: its decoder input is ``[BOS, ...phonemes, PAD...]`` with no
language slot, so its ``[BOS]`` is at position 0 in both regimes.

Two things are pinned here and they are different claims.

**Register.** The seed is the same two positions training uses, so the decoder starts where it
was taught to start. Measured on a model trained to memorise a closed 8-word list: a lone
``[BOS]`` seed reproduced 5 of 8 and derailed on two words entirely, `table` coming out
`telephant`; seeding the training prefix reproduces 8 of 8.

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

from bridge.domain.datamodels import BridgeInput, ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.model import WRITES_ORTH
from bridge.domain.tokenizer import BridgeTokenizer

WORDS = ["long", "pencil", "hello world"]
# The most tokens a row may write after its start. These tests read the start, so it is small.
CAP = 6


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


@pytest.fixture(scope="module")
def inputs(tokenizer):
    return tokenizer.encode_input(WORDS)


@pytest.fixture(scope="module")
def start(tokenizer):
    """The start for rows with no language given, like ``WORDS``."""
    return tokenizer.orth_start([None] * len(WORDS))


@pytest.mark.parametrize("pathway", sorted(WRITES_ORTH))
def test_generation_opens_with_the_same_prefix_training_uses(
    tokenizer, model, inputs, start, pathway
):
    """The claim, stated as the equality it is.

    Not "starts with a language token then BOS", which a hand-built seed would also satisfy
    while drifting from the tokenizer. The generated prefix must equal the first two ids
    the tokenizer's own target for these words feeds the decoder, so the two cannot
    diverge.
    """
    target = tokenizer.encode_target(WORDS)
    assert target is not None
    expected_prefix = target.orth.ids[:, :2]
    assert torch.equal(start, expected_prefix), "orth_start must be the training prefix"

    result = model.generate(
        inputs, pathway, max_new_tokens=CAP, orth_start=start, deterministic=True
    )

    assert torch.equal(result.orth_tokens[:, :2], expected_prefix), (
        f"{pathway} opened with {result.orth_tokens[:, :2].tolist()}, "
        f"training uses {expected_prefix.tolist()}"
    )


def test_the_prefix_is_the_language_the_caller_asked_for(tokenizer, model):
    """Conditioning, at the level this test can see it: the seed carries the caller's choice.

    Three languages over one input, so a seed that ignored the caller and hardcoded one
    token, or took the input's language, would fail: the input is English every time.
    Whether the trained model then *uses* the token is a question about training, measured
    in decision record 0008 rather than here, because it needs a trained model and a
    multilingual corpus.
    """
    english = tokenizer.encode_input(["no"], ["EN"])
    seen = {}
    for language in (None, "EN", "ES"):
        result = model.generate(
            english,
            "p2o",
            max_new_tokens=CAP,
            orth_start=tokenizer.orth_start([language]),
            deterministic=True,
        )
        seen[language] = names(tokenizer, result.orth_tokens[0, :2])

    assert seen == {
        None: ["--", "[BOS]"],
        "EN": ["EN", "[BOS]"],
        "ES": ["ES", "[BOS]"],
    }, seen


def test_a_mixed_row_is_seeded_with_its_first_words_language(tokenizer, model):
    """The seed is two positions, so it names one language: the first word's.

    Oracle: hand-written, and the target encoding. "no me" tagged ``ES EN`` and ``EN ES``
    differ only in which language comes first, and the decoder is fed that one first. A
    later change of language is the decoder's to emit; a seed naming the last word's
    language, or the row's most common one, would differ from what training feeds.
    """
    texts, languages = ["no me", "no me"], ["ES EN", "EN ES"]
    inputs = tokenizer.encode_input(texts, languages)
    orth_start = tokenizer.orth_start(["ES", "EN"])
    assert torch.equal(orth_start, tokenizer.encode_target(texts, languages).orth.ids[:, :2])

    for pathway in sorted(WRITES_ORTH):
        result = model.generate(
            inputs, pathway, max_new_tokens=CAP, orth_start=orth_start, deterministic=True
        )
        assert [names(tokenizer, row[:2]) for row in result.orth_tokens] == [
            ["ES", "[BOS]"],
            ["EN", "[BOS]"],
        ], pathway


def test_an_input_with_no_orthography_is_spelled_from_the_start_it_is_given(tokenizer, model):
    """``p2o`` is the one pathway that writes orthography without reading any.

    Its input can carry phonology alone, and then nothing orthographic comes from the
    input at all: the first two tokens are the start the caller passed, for each row.
    """
    phonology_only = BridgeInput(phon=tokenizer.encode_input(WORDS).phon)
    assert phonology_only.orth is None

    orth_start = tokenizer.orth_start([None, "ES", "EN"])
    result = model.generate(
        phonology_only, "p2o", max_new_tokens=CAP, orth_start=orth_start, deterministic=True
    )

    assert [names(tokenizer, row[:2]) for row in result.orth_tokens] == [
        ["--", "[BOS]"],
        ["ES", "[BOS]"],
        ["EN", "[BOS]"],
    ]


@pytest.mark.parametrize("pathway", sorted(WRITES_ORTH))
def test_the_probability_history_stays_aligned_with_the_tokens(
    tokenizer, model, inputs, start, pathway
):
    """One distribution per emitted token, including the seeded ones.

    A single row certain of BOS, under a two-token prefix, would leave the histories off
    by one, which is invisible unless something counts them. Each seeded row must be
    certain of the token actually seeded there, so this also catches a row built for the
    wrong token.
    """
    result = model.generate(
        inputs, pathway, max_new_tokens=CAP, orth_start=start, deterministic=True
    )

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


def test_seeding_the_decoder_directly_without_a_start_is_refused(model, inputs):
    """``_generate`` is reachable on its own, and a missing start must say so.

    Silently falling back to a lone ``[BOS]`` would give any caller that bypasses
    ``generate`` a decoder started from a state it was never trained on.
    """
    with pytest.raises(ValueError, match="needs orth_start"):
        model._generate(
            pathway="p2o",
            phon_enc_input=inputs.phon.ids,
            phon_enc_pad_mask=inputs.phon.pad_mask,
            max_new_tokens=CAP,
            deterministic=True,
        )
