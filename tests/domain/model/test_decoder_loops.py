"""Pins the batched rewrite of the two autoregressive decoder loops (issue #221).

Both loops used to run a Python ``for b in range(batch_size)`` every step to append
per-item values, and both re-embedded their entire prefix each step. They now stay dense
and batched, embed only the position they just generated, and cut the ragged
``GenerationOutput`` shapes once at the end.

Two invariants survive that rewrite only if it is done correctly, and neither is covered
elsewhere:

* ``orth_probs`` is *genuinely ragged*: a sequence stops collecting distributions at its
  own EOS. Every other generation test drives identical batch items, where a misaligned
  split is invisible because all the lengths coincide.
* the phonological loop must embed one position per step at the right position offset.
  Getting the offset wrong still produces plausible output, just from the wrong slot of
  the position table.
"""

import pytest
import torch

from bridge.domain.datamodels import ModelConfig
from bridge.domain.model import Model
from tests.vocab import TEST_VOCAB

D_MODEL = 16
BATCH = 4
NON_EOS_TOKEN = 7  # any orthographic id that is not [EOS]


@pytest.fixture
def model():
    model = Model(ModelConfig(vocab=TEST_VOCAB, d_model=D_MODEL, nhead=2, seed=5))
    model.eval()
    return model


@pytest.fixture
def prompt():
    """A fixed encoder memory, identical for every batch item.

    Identical items mean every item sees the same distribution at the same step, which is
    what lets the ragged-split assertions below compare rows across items.
    """
    return torch.linspace(-1, 1, D_MODEL).expand(BATCH, 1, D_MODEL).contiguous()


def run_orth_loop(model, prompt, finish_at):
    """Drive the orthographic loop with sampling forced to emit EOS on a schedule.

    ``finish_at`` maps batch index -> the step at which that item first emits EOS. Items
    absent from the map never finish.
    """
    steps = {"n": 0}

    def forced_sample(probs, deterministic):
        step = steps["n"]
        steps["n"] += 1
        tokens = torch.full((BATCH, 1), NON_EOS_TOKEN, dtype=torch.long)
        for b, when in finish_at.items():
            if step == when:
                tokens[b, 0] = TEST_VOCAB.orth_eos_id
        return tokens

    model.ortho_sample = forced_sample
    tokens = torch.full((BATCH, 1), TEST_VOCAB.orth_bos_id, dtype=torch.long)
    with torch.no_grad():
        probs, out_tokens = model.orthography_decoder_loop(
            model.generate_triangular_mask(model.max_orth_seq_len),
            model.embed_orth_tokens(tokens),
            tokens,
            prompt,
            deterministic=True,
        )
    return probs, out_tokens, steps["n"]


@pytest.mark.parametrize(
    ("finish_at", "expected_lengths"),
    [
        ({0: 0, 1: 2, 2: 5}, [2, 4, 7, 30]),  # mixed, one item never finishing
        ({}, [30, 30, 30, 30]),  # nobody finishes: one row per step, plus BOS
        ({0: 0, 1: 0, 2: 0, 3: 0}, [2, 2, 2, 2]),  # all finish at once, loop breaks early
        ({0: 3, 1: 3, 2: 3, 3: 28}, [5, 5, 5, 30]),  # three wait on a straggler
    ],
)
def test_orth_probs_stop_at_each_sequences_own_eos(model, prompt, finish_at, expected_lengths):
    """An item finishing first at step ``k`` keeps ``k + 2`` rows: BOS, then steps 0..k."""
    probs, _, _ = run_orth_loop(model, prompt, finish_at)
    assert [len(p) for p in probs] == expected_lengths


def test_orth_probs_rows_stay_aligned_to_their_step(model, prompt):
    """The ragged cut must be a prefix per item, not a shifted or gathered slice.

    Every batch item shares a prompt, so at any given step they all see the same
    distribution. Item 0 keeps only its first row, item 3 keeps all of them, so if the
    split were misaligned, the rows they share would disagree.
    """
    probs, _, _ = run_orth_loop(model, prompt, {0: 0, 1: 2, 2: 5})

    for b, row in enumerate(probs):
        assert torch.equal(row[0], probs[0][0]), f"item {b} has a different BOS placeholder"
        assert row[0][TEST_VOCAB.orth_bos_id] == 1
        for step, kept in enumerate(row[1:]):
            assert torch.equal(kept, probs[3][1 + step]), f"item {b} step {step} misaligned"
            assert kept.sum().item() == pytest.approx(1.0, abs=1e-5)


def seed_multihot(model):
    """The (batch, 1, V) [BOS] seed the phonological loop starts from."""
    multihot = torch.zeros((BATCH, 1, model.phonological_vocabulary_size), dtype=torch.long)
    multihot[:, 0, TEST_VOCAB.phon_bos_id] = 1
    return multihot


def fixed_schedule(model):
    """Sampling forced to a known, EOS-free feature per step, so the loop runs to length."""
    step = {"n": 0}

    def sample(last_token_probs, deterministic):
        vocab_size = model.phonological_vocabulary_size
        presence = torch.zeros((BATCH, vocab_size - 1), dtype=torch.long)
        presence[:, step["n"] % 5] = 1  # features 0-4 are real phonetic features, never [EOS]
        step["n"] += 1
        embedding_input = torch.zeros((BATCH, vocab_size), dtype=torch.long)
        embedding_input[:, : vocab_size - 1] = presence
        return presence, embedding_input

    return sample


def test_incremental_embedding_equals_embedding_the_whole_prefix(model, prompt):
    """The invariant behind embedding one position at a time instead of the whole prefix.

    A wrong position offset still produces plausible output, drawn from the wrong slot of
    the position table, so the only assertion that catches it is comparing the decoder
    input the loop actually built against embedding the same prefix in one call.
    """
    multihot = seed_multihot(model)
    model.phono_sample = fixed_schedule(model)

    fed = []
    model.phonology_decoder.register_forward_pre_hook(lambda _, args: fed.append(args[0]))

    with torch.no_grad():
        _, _, phon_tokens = model.phonology_decoder_loop(
            model.generate_triangular_mask(model.max_phon_seq_len),
            model.embed_phon_vectors(multihot),
            multihot,
            prompt,
            deterministic=True,
        )

    built = fed[-1]
    assert built.shape[1] > 1, "the loop must actually have appended rows to embed"

    # Rebuild the buffer the loop accumulated, and embed all of it in one call.
    full = torch.zeros(
        (BATCH, len(phon_tokens[0]), model.phonological_vocabulary_size), dtype=torch.long
    )
    for b, item in enumerate(phon_tokens):
        for position, features in enumerate(item):
            full[b, position, features] = 1
    with torch.no_grad():
        expected = model.embed_phon_vectors(full[:, : built.shape[1]])

    assert torch.allclose(built, expected, atol=1e-6)


def test_the_loop_embeds_a_linear_number_of_positions(model, prompt):
    """Re-embedding the prefix every step is what made generation quadratic in length.

    Asserted in aggregate, so a rewrite that batches or reorders the embedding work still
    passes and only a genuine return to the quadratic path fails.
    """
    multihot = seed_multihot(model)
    model.phono_sample = fixed_schedule(model)

    embedded_rows = []
    embed = model.embed_phon_vectors
    model.embed_phon_vectors = lambda v, position_offset=0: (
        embedded_rows.append(v.shape[1]) or embed(v, position_offset)
    )

    with torch.no_grad():
        _, _, phon_tokens = model.phonology_decoder_loop(
            model.generate_triangular_mask(model.max_phon_seq_len),
            embed(multihot),
            multihot,
            prompt,
            deterministic=True,
        )

    steps = len(phon_tokens[0]) - 1  # the seed row is not a step
    quadratic = steps * (steps + 1) // 2
    assert steps > 5, "need enough steps for the two growth rates to differ"
    assert sum(embedded_rows) <= 2 * steps, (
        f"embedded {sum(embedded_rows)} positions over {steps} steps; "
        f"re-embedding the prefix would be ~{quadratic}"
    )
