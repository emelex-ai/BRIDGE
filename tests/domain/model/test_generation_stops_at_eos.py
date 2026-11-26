"""A finished sequence stops contributing content (issue #231).

Both decoder loops run until *every* sequence in the batch has emitted `[EOS]`, because a
dense batch cannot stop early for one row. `CharacterTokenizer.decode` strips `[EOS]`
rather than stopping at it, so anything a finished row went on to sample would reach the
caller as part of the word.

The guard is the standard batched-generation one: once a row is finished it emits padding.
So the returned tensor is self-describing, content then `[EOS]` then padding, and `decode`
produces the right string as it is. ``docs/decisions/0009`` carries the measurements.

**Batch invariance is exact here, but only just, and only on this configuration.** Batching
perturbs `global_encoding` by up to about 6e-07, and `phono_sample` thresholds a probability
at `> 0.5`, so a feature sitting within an ulp of 0.5 can flip between a solo and a batched
run. That is unrelated to padding, and it has been observed elsewhere in the lexicon. It
does not reach these tests: measured in this exact configuration (seed 18, `MIXED`, a cap
of 28, CPU), the smallest `|P(ON) - 0.5|` is 5.4e-04, about 900 times the perturbation, and
the smallest orthographic argmax margin is 6.7e-05, about 110 times it. The seed was chosen
by that measurement, out of 40: it is one of three on which `p2o` ends two rows at
different steps, and of those it has the widest orthographic margin. If a future edit to
the model config or the word list makes an invariance test below fail by a single feature
or a single character, suspect that knife edge before suspecting the padding guard.

**The oracle throughout is batch invariance.** A causal decoder's rows are independent, so
what a word generates must not depend on which other words share its batch. That is a
property of the system rather than of this implementation, it holds with or without the
guard for the content that precedes `[EOS]`, and it is what distinguishes "stopped emitting"
from "changed what it emits". Tests that only assert padding-after-`[EOS]` would pass
against a guard that quietly disturbed the real output too.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.model import WRITES_ORTH
from bridge.domain.tokenizer import BridgeTokenizer

# Chosen so the batch terminates raggedly rather than in lockstep: a lockstep batch has no
# post-termination positions and so cannot see this defect at all.
MIXED = ["a", "cat", "elephant", "hi", "computer"]

# The most tokens a row may write after its start. A row that never emits [EOS] writes
# exactly this many.
CAP = 28


@pytest.fixture(scope="module")
def tokenizer():
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def vocab(tokenizer):
    return VocabSpec.from_tokenizer(tokenizer)


@pytest.fixture(scope="module")
def model(vocab):
    built = Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=18))
    built.eval()
    return built


def orth_rows(result):
    return [[int(v) for v in row] for row in result.orth_tokens]


def first_index(row, value):
    return next((i for i, v in enumerate(row) if v == value), None)


def phon_feature_sets(result, item):
    """The feature indices active at each position, as sorted tuples."""
    return [tuple(sorted(int(x) for x in t.reshape(-1).tolist())) for t in result.phon_tokens[item]]


def phon_first_eos(sets, eos_id):
    return next((i for i, features in enumerate(sets) if eos_id in features), None)


# An untrained model terminates wherever it happens to. On this word list only `p2o` ends
# a row early, two of its five, so a test written against it cannot choose where a row
# ends, and two of three pathways see nothing. Driving the sampler directly puts the
# termination schedule under the test's control, which is the only way to cover all three
# shapes deliberately.
#
# The sampler is an input here, not an oracle. What is asserted is the loop's contract given a
# known emission schedule, and the schedule is chosen so a loop that padded on the wrong
# condition produces a visibly different answer.


class ScriptedOrthoSampler:
    """Emit a fixed token per item per step, so `[EOS]` lands exactly where the test wants."""

    def __init__(self, plan, device):
        self.plan = plan
        self.device = device
        self.step = 0

    def __call__(self, probs, deterministic):
        column = [row[self.step] if self.step < len(row) else row[-1] for row in self.plan]
        self.step += 1
        return torch.tensor(column, dtype=torch.long, device=self.device).unsqueeze(-1)


def generate(model, tokenizer, words, pathway, deterministic=True):
    """Generate from ``words``, which carry no language, starting orthography unspecified."""
    return model.generate(
        tokenizer.encode_input(words),
        pathway,
        max_new_tokens=CAP,
        orth_start=tokenizer.orth_start([None] * len(words)),
        deterministic=deterministic,
    )


def run_scripted_orth(model, tokenizer, plan, words):
    real = model.ortho_sample
    model.ortho_sample = ScriptedOrthoSampler(plan, model.device)
    try:
        result = generate(model, tokenizer, words, "p2o")
    finally:
        model.ortho_sample = real
    return orth_rows(result), result


def test_the_loop_pads_every_row_from_its_own_eos(tokenizer, model, vocab):
    """Three shapes at once: an early finisher, a late one, and one that never terminates.

    Written as an explicit expected row per item rather than a property, so a loop that
    padded from the wrong step, or from the step some *other* row finished, cannot satisfy it.
    """
    a, b, c = (tokenizer.char_tokenizer.char_2_idx[ch] for ch in "abc")
    eos, pad = vocab.orth_eos_id, vocab.orth_pad_id
    filler = tokenizer.char_tokenizer.char_2_idx["z"]

    # item 0 finishes at step 1, item 1 at step 4, item 2 never.
    plan = [
        [a, eos] + [filler] * 30,
        [a, b, c, a, eos] + [filler] * 30,
        [c] * 32,
    ]
    rows, _ = run_scripted_orth(model, tokenizer, plan, ["cat", "dog", "hat"])

    width = len(rows[0])
    prefix = [int(v) for v in tokenizer.orth_start([None])[0]]
    # The loop stops when every row has finished or at the cap, so item 2, which never
    # finishes, sets the width: the two tokens of the start and CAP more.
    assert width == len(prefix) + CAP, width

    assert rows[0] == prefix + [a, eos] + [pad] * (width - len(prefix) - 2)
    assert rows[1] == prefix[:1] + [prefix[1], a, b, c, a, eos] + [pad] * (width - len(prefix) - 5)
    assert rows[2] == prefix + [c] * (width - len(prefix)), (
        "a row that never emits [EOS] must keep all of its content and gain no padding"
    )


def test_a_lockstep_batch_gains_no_padding_at_all(tokenizer, model, vocab):
    """Rows that finish on the same step need no suppression.

    If padding appeared here the guard would be firing on the loop ending rather than on each
    row's own termination.
    """
    a = tokenizer.char_tokenizer.char_2_idx["a"]
    eos, pad = vocab.orth_eos_id, vocab.orth_pad_id

    rows, _ = run_scripted_orth(
        model, tokenizer, [[a, a, eos] + [a] * 30] * 3, ["cat", "dog", "hat"]
    )

    assert rows[0] == rows[1] == rows[2]
    assert pad not in rows[0], f"padding appeared in a lockstep batch: {rows[0]}"
    assert rows[0][-1] == eos, "the batch should stop the step after everyone terminates"


def test_padding_starts_after_the_terminator_not_at_it(tokenizer, model, vocab):
    """Masking one step early would delete the `[EOS]` and leave a sequence that never
    says it finished, which reads as correct once decoded and is not."""
    a = tokenizer.char_tokenizer.char_2_idx["a"]
    eos = vocab.orth_eos_id

    rows, _ = run_scripted_orth(model, tokenizer, [[a, eos] + [a] * 30, [a] * 32], ["cat", "dog"])

    assert rows[0].count(eos) == 1, f"expected exactly one [EOS], got {rows[0]}"
    assert rows[0][3] == eos, f"[EOS] should sit at position 3, row is {rows[0]}"
    assert set(rows[0][4:]) == {vocab.orth_pad_id}


def test_nothing_but_padding_follows_the_orthographic_eos(tokenizer, model, vocab):
    """The same contract through the public path, on the pathway that terminates naturally.

    Only `p2o` ends early on this word list. The scripted tests above cover the pathways and
    shapes that an untrained model does not happen to produce.
    """

    rows = orth_rows(generate(model, tokenizer, MIXED, "p2o"))

    terminated = 0
    for item, row in enumerate(rows):
        eos = first_index(row, vocab.orth_eos_id)
        if eos is None:
            continue
        terminated += 1
        after = row[eos + 1 :]
        assert set(after) <= {vocab.orth_pad_id}, (
            f"item {item} emitted {[v for v in after if v != vocab.orth_pad_id]} "
            f"after its [EOS] at position {eos}"
        )
    assert terminated >= 2, (
        f"only {terminated} of {len(rows)} items terminated, so this batch barely sees the "
        "defect; pick words whose generations end at different steps"
    )


def test_the_orthographic_eos_itself_survives(tokenizer, model, vocab):
    """One terminator per row, through the public path."""

    rows = orth_rows(generate(model, tokenizer, MIXED, "p2o"))

    assert any(vocab.orth_eos_id in row for row in rows), "no [EOS] survived anywhere"
    for item, row in enumerate(rows):
        assert row.count(vocab.orth_eos_id) <= 1, (
            f"item {item} contains {row.count(vocab.orth_eos_id)} [EOS] tokens; everything "
            "after the first must be padding"
        )


@pytest.mark.parametrize("pathway", sorted(WRITES_ORTH))
def test_orthographic_content_does_not_depend_on_the_rest_of_the_batch(
    tokenizer, model, vocab, pathway
):
    """The real oracle: a causal decoder's rows are independent.

    Each word is generated alone and then in a batch, and the content up to and including its
    `[EOS]` must be identical. It pins that suppressing post-termination output does not
    disturb the output that precedes it. It runs on all three orthographic pathways because
    it needs no row to terminate.
    """
    batched = orth_rows(generate(model, tokenizer, MIXED, pathway))

    for item, word in enumerate(MIXED):
        alone = orth_rows(generate(model, tokenizer, [word], pathway))[0]
        cut_alone = alone[: (first_index(alone, vocab.orth_eos_id) or len(alone) - 1) + 1]
        row = batched[item]
        cut_batch = row[: (first_index(row, vocab.orth_eos_id) or len(row) - 1) + 1]
        assert cut_batch == cut_alone, (
            f"{pathway} {word!r}: generated {cut_batch} in a batch but {cut_alone} alone, so "
            "batching changed the content, not just what follows it"
        )


def test_decoding_a_batch_returns_what_the_model_generated(tokenizer, model, vocab):
    """The symptom of issue #231, on the user-facing path.

    ``decode`` strips padding, so a correctly padded row decodes to the right string. The
    comparison is against an independent truncation computed here, not against ``decode``
    called a second way.
    """
    rows = orth_rows(generate(model, tokenizer, MIXED, "p2o"))

    def upto_eos(row):
        eos = first_index(row, vocab.orth_eos_id)
        return row if eos is None else row[:eos]

    assert tokenizer.char_tokenizer.decode(rows) == tokenizer.char_tokenizer.decode(
        [upto_eos(row) for row in rows]
    )


def test_a_word_decodes_the_same_alone_as_in_a_batch(tokenizer, model):
    """Batch invariance again, at the level a caller actually observes.

    The defect: a word decodes correctly alone and incorrectly beside a longer word, which
    is the worst shape a defect can take: correct in the small case someone debugs with.
    """
    batched = tokenizer.char_tokenizer.decode(orth_rows(generate(model, tokenizer, MIXED, "p2o")))
    for item, word in enumerate(MIXED):
        alone = tokenizer.char_tokenizer.decode(
            orth_rows(generate(model, tokenizer, [word], "p2o"))
        )[0]
        assert batched[item] == alone, (
            f"{word!r} decoded {batched[item]!r} batched, {alone!r} alone"
        )


def test_the_probability_history_still_marks_where_content_ends(tokenizer, model, vocab):
    """``orth_probs`` is cut per item at its `[EOS]`, and the token row agrees with it.

    Tokens that ran on past where the probabilities stop would leave the returned object
    internally inconsistent.
    """
    result = generate(model, tokenizer, MIXED, "p2o")
    rows = orth_rows(result)

    for item, row in enumerate(rows):
        eos = first_index(row, vocab.orth_eos_id)
        expected = len(row) if eos is None else eos + 1
        assert len(result.orth_probs[item]) == expected, (
            f"item {item}: {len(result.orth_probs[item])} probability rows for {expected} "
            "real token positions"
        )


def test_nothing_but_padding_follows_the_phonological_eos(tokenizer, model, vocab):
    """The same defect on the phonological loop.

    Stochastic sampling with a fixed seed, because greedy decoding on an untrained model
    terminates every item on the same step and a lockstep batch cannot show this. A padded
    phonological position is the one-hot at ``phon_pad_id``, which is the encoding
    ``phono_sample`` already uses for an all-off vector rather than a new convention.
    """
    torch.manual_seed(3)
    result = generate(model, tokenizer, MIXED * 3, "o2p", deterministic=False)

    terminated = 0
    for item in range(len(result.phon_tokens)):
        sets = phon_feature_sets(result, item)
        eos = phon_first_eos(sets, vocab.phon_eos_id)
        if eos is None:
            continue
        terminated += 1
        for position, features in enumerate(sets[eos + 1 :], start=eos + 1):
            assert features == (vocab.phon_pad_id,), (
                f"phonological item {item} position {position} holds {features} after its "
                f"[EOS] at {eos}"
            )
    assert terminated, "no phonological item terminated, so this batch cannot see the defect"


def test_phonological_content_does_not_depend_on_the_rest_of_the_batch(tokenizer, model, vocab):
    """Batch invariance for phonology, up to and including each item's `[EOS]`."""
    batched = generate(model, tokenizer, MIXED, "o2p")

    for item, word in enumerate(MIXED):
        alone = generate(model, tokenizer, [word], "o2p")
        alone_sets = phon_feature_sets(alone, 0)
        batch_sets = phon_feature_sets(batched, item)
        cut = phon_first_eos(alone_sets, vocab.phon_eos_id)
        end = len(alone_sets) if cut is None else cut + 1
        assert batch_sets[:end] == alone_sets[:end], (
            f"{word!r}: phonological content differs between a batch and a solo run"
        )


def test_phonological_padding_uses_the_existing_all_off_encoding(tokenizer, model, vocab):
    """The padded row must be the convention the sampler already uses, not a new one.

    ``phono_sample`` switches on ``phon_pad_id`` for a vector with every feature off, so a
    suppressed position is indistinguishable in form from a legitimately empty one. Anything
    else would need every downstream consumer to learn a second way of saying "nothing here".
    """
    torch.manual_seed(3)
    result = generate(model, tokenizer, MIXED * 3, "o2p", deterministic=False)

    padded = [
        features
        for item in range(len(result.phon_tokens))
        for features in phon_feature_sets(result, item)
        if features == (vocab.phon_pad_id,)
    ]
    assert padded, "no padded phonological position was produced, so this asserts nothing"
    assert all(len(features) == 1 for features in padded)
