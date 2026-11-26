"""``max_new_tokens`` is the only thing that ends a sequence the model does not end itself.

No length is built into the model, so ``Model.generate`` takes the cap from its caller:
``max_new_tokens``, keyword-only and required. The contract, each line pinned below:

* It is **required**. There is no default to fall back on, and omitting it raises.
* It counts tokens written **after the start**: ``[LANG, BOS]`` for orthography and
  ``[BOS]`` for phonology. A cap of ``n`` gives a row of at most ``2 + n`` orthographic
  tokens, or ``1 + n`` phonological positions.
* A row **stops at** ``[EOS]`` **or at the cap**, whichever comes first.
* A row that reaches the cap is **cut and has no** ``[EOS]``, which is how a caller tells
  a cut row from a finished one.
* A row that finishes first **emits padding** from then on, as ``docs/decisions/0009``
  set out, and the loop stops once every row has finished.
* On ``op2op`` the one cap applies to **each** modality written.

**The oracle is analytic throughout.** The samplers are scripted, so the step at which
each row emits ``[EOS]`` is chosen by the test and every expected row is written out by
hand from the contract. Nothing here depends on what an untrained model happens to emit.
The sampler is an input, not the thing under test: the loops, the cap and the padding
guard are real.
"""

import inspect

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.datamodels.pathways import PATHWAY_IO, PATHWAYS
from bridge.domain.model import Model

WORDS = ["cat", "dog", "hat"]
# The orthographic token and the phonological feature a scripted row emits until it ends.
LETTER = "z"
FEATURE = 0


@pytest.fixture(scope="module")
def vocab(shared_tokenizer):
    return VocabSpec.from_tokenizer(shared_tokenizer)


@pytest.fixture(scope="module")
def model(vocab):
    return Model(ModelConfig(vocab=vocab, d_model=16, nhead=2, seed=5)).eval()


@pytest.fixture(scope="module")
def inputs(shared_tokenizer):
    return shared_tokenizer.encode_input(WORDS)


@pytest.fixture(scope="module")
def start(shared_tokenizer):
    return shared_tokenizer.orth_start([None] * len(WORDS))


def scripted(monkeypatch, model, shared_tokenizer, eos_at):
    """Make both samplers emit ``[EOS]`` for row ``b`` at step ``eos_at[b]``, counted from
    zero, and never for a row whose entry is ``None``. Returns the number of steps each
    loop took, filled in as the loops run."""
    vocab = model.model_config.vocab
    letter = shared_tokenizer.char_tokenizer.char_2_idx[LETTER]
    steps = {"orth": 0, "phon": 0}

    def ortho_sample(probs, deterministic):
        step = steps["orth"]
        steps["orth"] += 1
        return torch.tensor([[vocab.orth_eos_id if at == step else letter] for at in eos_at])

    def phono_sample(probs, deterministic):
        step = steps["phon"]
        steps["phon"] += 1
        presence = torch.zeros((len(eos_at), vocab.phon_vocab_size - 1), dtype=torch.long)
        for row, at in enumerate(eos_at):
            presence[row, vocab.phon_eos_id if at == step else FEATURE] = 1
        embedding_input = torch.zeros((len(eos_at), vocab.phon_vocab_size), dtype=torch.long)
        embedding_input[:, :-1] = presence
        return presence, embedding_input

    monkeypatch.setattr(model, "ortho_sample", ortho_sample)
    monkeypatch.setattr(model, "phono_sample", phono_sample)
    return steps


def orth_rows(result) -> list[list[int]]:
    return result.orth_tokens.tolist()


def phon_rows(result) -> list[list[tuple[int, ...]]]:
    """The active features at each position of each row, as sorted tuples."""
    return [[tuple(sorted(t.tolist())) for t in row] for row in result.phon_tokens]


def writes(pathway: str, modality: str) -> bool:
    return modality in PATHWAY_IO[pathway][1]


@pytest.mark.parametrize("method", ["generate", "_generate"])
def test_max_new_tokens_is_keyword_only_and_has_no_default(method):
    """Oracle: the signature. A default would be a length the model does not have, and it
    would cut output in silence for every caller who did not think to set it."""
    parameter = inspect.signature(getattr(Model, method)).parameters["max_new_tokens"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


@pytest.mark.parametrize("pathway", PATHWAYS)
def test_generating_without_a_cap_raises_naming_it(model, inputs, start, pathway):
    """The same rule from the caller's side, on every pathway, with everything else about
    the call right. The control is that call with a cap, which runs."""
    with pytest.raises(TypeError, match="max_new_tokens"):
        model.generate(inputs, pathway, orth_start=start, deterministic=True)

    model.generate(inputs, pathway, max_new_tokens=1, orth_start=start, deterministic=True)


@pytest.mark.parametrize("bad_cap", [0, -3, 2.0, "4", None, True])
def test_a_cap_that_is_not_a_positive_int_is_refused_and_the_model_left_as_it_was(
    model, inputs, start, bad_cap
):
    """Through the public call: the cap is checked before anything runs, so a model in
    train mode stays there. The sweep over every pathway is in
    ``test_validate_generate_input.py``."""
    model.train()
    try:
        with pytest.raises(ValueError, match="max_new_tokens must be a positive int"):
            model.generate(
                inputs, "op2op", max_new_tokens=bad_cap, orth_start=start, deterministic=True
            )
        assert model.training
    finally:
        model.eval()


@pytest.mark.parametrize("pathway", PATHWAYS)
@pytest.mark.parametrize("cap", [1, 2, 7])
def test_a_row_that_never_finishes_writes_exactly_the_cap_after_its_start(
    monkeypatch, model, shared_tokenizer, inputs, start, vocab, pathway, cap
):
    """Oracle: analytic. No row emits ``[EOS]``, so each is cut at the cap: the start, then
    ``cap`` tokens, and no ``[EOS]`` and no padding anywhere. On ``op2op`` both modalities
    get the whole cap.

    Counting the start against the cap would write two orthographic tokens too few, and at
    a cap of one or two nothing at all. A cap of one is the boundary: one step, one token.
    """
    steps = scripted(monkeypatch, model, shared_tokenizer, eos_at=[None] * len(WORDS))
    letter = shared_tokenizer.char_tokenizer.char_2_idx[LETTER]

    result = model.generate(
        inputs, pathway, max_new_tokens=cap, orth_start=start, deterministic=True
    )

    if writes(pathway, "orth"):
        assert orth_rows(result) == [[*row, *[letter] * cap] for row in start.tolist()]
        assert [len(history) for history in result.orth_probs] == [2 + cap] * len(WORDS)
    else:
        assert result.orth_tokens is None
    if writes(pathway, "phon"):
        assert phon_rows(result) == [[(vocab.phon_bos_id,), *[(FEATURE,)] * cap]] * len(WORDS)
        assert [len(history) for history in result.phon_probs] == [cap] * len(WORDS)
        assert [len(history) for history in result.phon_vecs] == [cap] * len(WORDS)
    else:
        assert result.phon_tokens is None
    assert steps == {m: cap if writes(pathway, m) else 0 for m in ("orth", "phon")}


def test_a_finished_row_pads_and_a_cut_row_does_not(
    monkeypatch, model, shared_tokenizer, inputs, start, vocab
):
    """Oracle: analytic, one expected row per item. With a cap of six, row 0 emits
    ``[EOS]`` at step 1, row 1 at step 4 and row 2 never.

    Rows 0 and 1 are their content, their ``[EOS]`` and padding to the width. Row 2 is cut:
    six tokens after the start, with no ``[EOS]`` and no padding. The ``[EOS]`` a row does
    emit is kept, and nothing is appended to the row that did not finish.
    """
    scripted(monkeypatch, model, shared_tokenizer, eos_at=[1, 4, None])
    z = shared_tokenizer.char_tokenizer.char_2_idx[LETTER]
    eos, pad = vocab.orth_eos_id, vocab.orth_pad_id

    result = model.generate(inputs, "op2op", max_new_tokens=6, orth_start=start, deterministic=True)

    lang, bos = start[0].tolist()
    assert orth_rows(result) == [
        [lang, bos, z, eos, pad, pad, pad, pad],
        [lang, bos, z, z, z, z, eos, pad],
        [lang, bos, z, z, z, z, z, z],
    ]
    # The probability history stops with the row: its start, then one step per token
    # up to and including its [EOS].
    assert [len(history) for history in result.orth_probs] == [4, 7, 8]

    f, p_bos = (FEATURE,), (vocab.phon_bos_id,)
    p_eos, p_pad = (vocab.phon_eos_id,), (vocab.phon_pad_id,)
    assert phon_rows(result) == [
        [p_bos, f, p_eos, p_pad, p_pad, p_pad, p_pad],
        [p_bos, f, f, f, f, p_eos, p_pad],
        [p_bos, f, f, f, f, f, f],
    ]


def test_an_eos_on_the_last_step_the_cap_allows_is_kept_and_one_step_later_is_never_reached(
    monkeypatch, model, shared_tokenizer, inputs, start, vocab
):
    """The boundary, from both sides. With a cap of four the steps are 0 to 3. Row 0 is
    scripted to end at step 3, the last one, and does. Row 1 is scripted to end at step 4,
    which the cap never reaches, so it is cut and holds no ``[EOS]``. Row 2 never ends."""
    steps = scripted(monkeypatch, model, shared_tokenizer, eos_at=[3, 4, None])
    z = shared_tokenizer.char_tokenizer.char_2_idx[LETTER]

    result = model.generate(inputs, "op2op", max_new_tokens=4, orth_start=start, deterministic=True)

    lang, bos = start[0].tolist()
    assert orth_rows(result) == [
        [lang, bos, z, z, z, vocab.orth_eos_id],
        [lang, bos, z, z, z, z],
        [lang, bos, z, z, z, z],
    ]
    f = (FEATURE,)
    assert phon_rows(result) == [
        [(vocab.phon_bos_id,), f, f, f, (vocab.phon_eos_id,)],
        [(vocab.phon_bos_id,), f, f, f, f],
        [(vocab.phon_bos_id,), f, f, f, f],
    ]
    assert steps == {"orth": 4, "phon": 4}


def test_the_loop_stops_when_every_row_has_finished_however_large_the_cap(
    monkeypatch, model, shared_tokenizer, inputs, start, vocab
):
    """A cap is a ceiling, not a length. Every row ends at step 2, so three steps are
    taken under a cap of two hundred, and the rows are three tokens past their start."""
    steps = scripted(monkeypatch, model, shared_tokenizer, eos_at=[2, 2, 2])
    z = shared_tokenizer.char_tokenizer.char_2_idx[LETTER]

    result = model.generate(
        inputs, "op2op", max_new_tokens=200, orth_start=start, deterministic=True
    )

    assert orth_rows(result) == [[*row, z, z, vocab.orth_eos_id] for row in start.tolist()]
    assert phon_rows(result) == [
        [(vocab.phon_bos_id,), (FEATURE,), (FEATURE,), (vocab.phon_eos_id,)]
    ] * len(WORDS)
    assert steps == {"orth": 3, "phon": 3}
