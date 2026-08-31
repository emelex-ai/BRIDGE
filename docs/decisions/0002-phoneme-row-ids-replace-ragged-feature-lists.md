# 0002. Phonology uses (batch, sequence) row ids, matching orthography

Status: Accepted
Date: 2026-08-30
Issue: #221

## Context

Phonological encoder and decoder inputs were `list[list[torch.Tensor]]`: an outer list per
batch item, an inner list per sequence position, each tensor holding the active phonetic
feature indices of one phoneme. Orthography was already a plain `(batch, sequence)` LongTensor.

The asymmetry cost real time and real code. Embedding meant a Python loop over batch by
position, averaging feature embeddings per phoneme, and `embed_phon_tokens` was measured at
17.4 percent of a CPU training step and 22.7 percent on CUDA. Every consumer needed a branch
for the ragged shape, and the two modalities could not share a validator, a decoder loop or a
container.

## Decision

Phonology carries `(batch, sequence)` integer tensors of **phoneme row ids**, structurally
identical to the orthographic side.

The enabling observation: a phoneme's embedding is the mean of its active feature embeddings,
which is **linear in the embedding weight**. So `A_norm @ W`, where `A_norm` is the
row-normalised phoneme-by-feature matrix, yields every phoneme's embedding in one small matmul,
independent of batch size. Embedding a batch becomes `F.embedding(rows, table)`.

The derived matrix is registered as a **non-persistent** buffer. It is a function of
`phonreps.csv`, so it stays out of `state_dict` and existing checkpoints keep loading under
`strict=True`.

This introduces a second integer id space. Row space, 0 to 90, is which phoneme. Feature space,
0 to 35, is which phonetic feature and remains what `GenerationOutput.phon_tokens`,
`PhonemeTokenizer.decode` and the loss targets use.

Rejected alternative: make phonemes a plain `nn.Embedding(91, d)`. Same runtime shape, but it
deletes the feature-sharing inductive bias that is the scientific premise of BRIDGE. Rejected
on modelling grounds, not performance.

## Evidence

Verified equivalent against a pristine worktree at `5d7e8d9`, over three model configs, six
word batches, all five pathways and all three modality filters:

| quantity | result |
|---|---|
| exact-valued (ids, masks, generated tokens, feature sets, decoded strings) | **27,064 of 27,064 identical** |
| initial weights across three configs | bitwise identical |
| float tensors including full-model gradients | worst deviation 8.9e-07 scale-normalised |
| loss trajectory over 12 Adam steps | within 2.07e-07 relative |

The control that makes those numbers readable: the same tree compared against itself under a
different BLAS thread count diverges by 2.229e-07, slightly more than the refactor does.
Reruns of an unchanged tree are bit-exact.

All CPU. CUDA reduction order differs and this says nothing about it.

## Consequences

One validator, one container and one shape serve both modalities. `bridge/core/phonreps.py`
went from seven exports to three as `PhonReps`, `load_phonreps` and `load_phonreps_array`
turned out to be shadows of `PhonemeTable`.

The cost is the second id space. Both spaces are `torch.long` and only naming separates them,
so a mix-up produces silently wrong output rather than an error. Mitigations: the parameter is
named `rows`, `_validate_phon_bounds` checks against the table row count rather than the
vocabulary size, and `Model.__init__` refuses a `VocabSpec` whose special-token ids disagree
with the table.

A `phonreps.csv` edit relabels every id with no shape change, so `VocabSpec` records a table
fingerprint and checkpoint loading warns on drift.
