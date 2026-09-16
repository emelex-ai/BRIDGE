"""A finished sequence stops contributing content (issue #231).

Both decoder loops run until *every* sequence in the batch has emitted `[EOS]`, because a
dense batch cannot stop early for one row. What they did with the rows that had already
finished was keep sampling and keep the result. `CharacterTokenizer.decode` strips `[EOS]`
rather than stopping at it, so that content reached the caller as part of the word.

Measured on an untrained model before the fix: a 5-word batch left 32 non-padding tokens
after a terminating `[EOS]`, and 4 of the 5 decoded to something longer than the model
produced. The phonological loop does the same thing, though it needs a batch whose items
terminate at different steps to show it: 8 items, 7 finishing at step 0 and one at step 4,
left 28 positions of sampled features after their own `[EOS]`.

The fix is the standard batched-generation guard: once a row is finished it emits padding.
So the returned tensor is self-describing, content then `[EOS]` then padding, and `decode`
produces the right string without being changed.

**Batch invariance is exact here, but only just, and only on this configuration.** Batching
perturbs `global_encoding` by up to about 1.2e-06, and `phono_sample` thresholds a probability
at `> 0.5`, so a feature sitting within an ulp of 0.5 can flip between a solo and a batched
run. That is pre-existing, unrelated to padding, and it has been observed elsewhere in the
lexicon. It does not reach these tests: measured in this exact configuration, the smallest
`|P(ON) - 0.5|` is 8.2e-03, about 6800 times the perturbation, and the smallest orthographic
argmax margin is 2.5e-05, about 21 times it. If a future edit to the model config or the word
list makes an invariance test below fail by a single feature or a single character, suspect
that knife edge before suspecting the padding guard.

**The oracle throughout is batch invariance.** A causal decoder's rows are independent, so
what a word generates must not depend on which other words share its batch. That is a
property of the system rather than of this implementation, it holds before and after the fix
for the content that precedes `[EOS]`, and it is what distinguishes "stopped emitting" from
"changed what it emits". Tests that only assert padding-after-`[EOS]` would pass against a
fix that quietly disturbed the real output too.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig, VocabSpec
from bridge.domain.model import Model
from bridge.domain.model.model import ORTH_DECODING
from bridge.domain.tokenizer import BridgeTokenizer

# Chosen so the batch terminates raggedly rather than in lockstep: a lockstep batch has no
# post-termination positions and so cannot see this defect at all.
MIXED = ["a", "cat", "elephant", "hi", "computer"]


@pytest.fixture(scope="module")
def tokenizer():
    return BridgeTokenizer()


@pytest.fixture(scope="module")
def vocab(tokenizer):
    return VocabSpec.from_tokenizer(tokenizer)


@pytest.fixture(scope="module")
def model(vocab):
    built = Model(ModelConfig(vocab=vocab, d_model=32, nhead=2, seed=5))
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


# ------------------------------------------------------- the loop's contract, scripted
#
# An untrained model terminates wherever it happens to. On this word list only `p2o` ends
# early at all, and no row runs to the position limit, so a test written against it can
# exercise neither the ragged case nor the never-terminating one, and two of three pathways
# see nothing. Driving the sampler directly puts the termination schedule under the test's
# control, which is the only way to cover all three shapes deliberately.
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


def run_scripted_orth(model, tokenizer, plan, words):
    """Run the real orthographic loop with a scripted sampler, returning the token rows."""
    encoding = tokenizer.encode(words)
    assert encoding is not None
    real = model.ortho_sample
    model.ortho_sample = ScriptedOrthoSampler(plan, model.device)
    try:
        result = model.generate(encoding, "p2o", deterministic=True)
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
    prefix = [int(v) for v in tokenizer.encode(["cat"]).orthographic.dec_input_ids[0, :2]]
    # The loop stops when every row has finished, so item 2 is what sets the width.
    assert width == model.max_orth_seq_len, width

    assert rows[0] == prefix + [a, eos] + [pad] * (width - len(prefix) - 2)
    assert rows[1] == prefix[:1] + [prefix[1], a, b, c, a, eos] + [pad] * (width - len(prefix) - 5)
    assert rows[2] == prefix + [c] * (width - len(prefix)), (
        "a row that never emits [EOS] must keep all of its content and gain no padding"
    )


def test_a_lockstep_batch_gains_no_padding_at_all(tokenizer, model, vocab):
    """Rows that finish on the same step need no suppression.

    If padding appeared here the guard would be firing on the loop ending rather than on each
    row's own termination, which is the mistake this whole change is about.
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


# --------------------------------------------------------------------- orthography, end to end


def test_nothing_but_padding_follows_the_orthographic_eos(tokenizer, model, vocab):
    """The same contract through the public path, on the pathway that terminates naturally.

    Only `p2o` ends early on this word list. The scripted tests above cover the pathways and
    shapes that an untrained model does not happen to produce.
    """
    encoding = tokenizer.encode(MIXED)
    assert encoding is not None

    rows = orth_rows(model.generate(encoding, "p2o", deterministic=True))

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
    encoding = tokenizer.encode(MIXED)
    assert encoding is not None

    rows = orth_rows(model.generate(encoding, "p2o", deterministic=True))

    assert any(vocab.orth_eos_id in row for row in rows), "no [EOS] survived anywhere"
    for item, row in enumerate(rows):
        assert row.count(vocab.orth_eos_id) <= 1, (
            f"item {item} contains {row.count(vocab.orth_eos_id)} [EOS] tokens; everything "
            "after the first must be padding"
        )


@pytest.mark.parametrize("pathway", sorted(ORTH_DECODING))
def test_orthographic_content_does_not_depend_on_the_rest_of_the_batch(
    tokenizer, model, vocab, pathway
):
    """The real oracle: a causal decoder's rows are independent.

    Each word is generated alone and then in a batch, and the content up to and including its
    `[EOS]` must be identical. This holds before the fix too, which is the point: it pins that
    suppressing post-termination output did not disturb the output that precedes it. It runs
    on all three orthographic pathways because it needs no row to terminate.
    """
    batched = orth_rows(model.generate(tokenizer.encode(MIXED), pathway, deterministic=True))

    for item, word in enumerate(MIXED):
        alone = orth_rows(model.generate(tokenizer.encode([word]), pathway, deterministic=True))[0]
        cut_alone = alone[: (first_index(alone, vocab.orth_eos_id) or len(alone) - 1) + 1]
        row = batched[item]
        cut_batch = row[: (first_index(row, vocab.orth_eos_id) or len(row) - 1) + 1]
        assert cut_batch == cut_alone, (
            f"{pathway} {word!r}: generated {cut_batch} in a batch but {cut_alone} alone, so "
            "batching changed the content, not just what follows it"
        )


def test_decoding_a_batch_returns_what_the_model_generated(tokenizer, model, vocab):
    """The reported symptom, on the user-facing path.

    ``decode`` is not changed by this fix. It strips padding already, so a correctly padded
    row decodes to the right string. The comparison is against an independent truncation
    computed here, not against ``decode`` called a second way.
    """
    encoding = tokenizer.encode(MIXED)
    assert encoding is not None
    rows = orth_rows(model.generate(encoding, "p2o", deterministic=True))

    def upto_eos(row):
        eos = first_index(row, vocab.orth_eos_id)
        return row if eos is None else row[:eos]

    assert tokenizer.char_tokenizer.decode(rows) == tokenizer.char_tokenizer.decode(
        [upto_eos(row) for row in rows]
    )


def test_a_word_decodes_the_same_alone_as_in_a_batch(tokenizer, model):
    """Batch invariance again, at the level a caller actually observes.

    Before the fix a word decoded correctly alone and incorrectly beside a longer word, which
    is the worst shape a defect can take: correct in the small case someone debugs with.
    """
    batched = tokenizer.char_tokenizer.decode(
        orth_rows(model.generate(tokenizer.encode(MIXED), "p2o", deterministic=True))
    )
    for item, word in enumerate(MIXED):
        alone = tokenizer.char_tokenizer.decode(
            orth_rows(model.generate(tokenizer.encode([word]), "p2o", deterministic=True))
        )[0]
        assert batched[item] == alone, (
            f"{word!r} decoded {batched[item]!r} batched, {alone!r} alone"
        )


def test_the_probability_history_still_marks_where_content_ends(tokenizer, model, vocab):
    """``orth_probs`` was already cut per item; the token row now agrees with it.

    Before the fix these disagreed: the probabilities stopped at `[EOS]` and the tokens ran
    on. Their agreeing is what makes the returned object internally consistent.
    """
    encoding = tokenizer.encode(MIXED)
    assert encoding is not None
    result = model.generate(encoding, "p2o", deterministic=True)
    rows = orth_rows(result)

    for item, row in enumerate(rows):
        eos = first_index(row, vocab.orth_eos_id)
        expected = len(row) if eos is None else eos + 1
        assert len(result.orth_probs[item]) == expected, (
            f"item {item}: {len(result.orth_probs[item])} probability rows for {expected} "
            "real token positions"
        )


# --------------------------------------------------------------------- phonology


def test_nothing_but_padding_follows_the_phonological_eos(tokenizer, model, vocab):
    """The same defect on the phonological loop.

    Stochastic sampling with a fixed seed, because greedy decoding on an untrained model
    terminates every item on the same step and a lockstep batch cannot show this. A padded
    phonological position is the one-hot at ``phon_pad_id``, which is the encoding
    ``phono_sample`` already uses for an all-off vector rather than a new convention.
    """
    encoding = tokenizer.encode(MIXED * 3)
    assert encoding is not None

    torch.manual_seed(3)
    result = model.generate(encoding, "o2p", deterministic=False)

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
    batched = model.generate(tokenizer.encode(MIXED), "o2p", deterministic=True)

    for item, word in enumerate(MIXED):
        alone = model.generate(tokenizer.encode([word]), "o2p", deterministic=True)
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
    encoding = tokenizer.encode(MIXED * 3)
    assert encoding is not None
    torch.manual_seed(3)
    result = model.generate(encoding, "o2p", deterministic=False)

    padded = [
        features
        for item in range(len(result.phon_tokens))
        for features in phon_feature_sets(result, item)
        if features == (vocab.phon_pad_id,)
    ]
    assert padded, "no padded phonological position was produced, so this asserts nothing"
    assert all(len(features) == 1 for features in padded)
