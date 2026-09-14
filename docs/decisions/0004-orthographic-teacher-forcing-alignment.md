# 0004. Orthographic teacher forcing targets the encoder sequence shifted by one

Status: Accepted
Date: 2026-08-30

## Context

`TrainingPipeline.compute_loss` scored the orthographic decoder against
`orthography.enc_input_ids[:, 2:]`, which is one position shorter than the decoder produces.
`CrossEntropyLoss` raised on every batch, so `p2o` and `op2op` could not complete a single
training step. Only `o2p` and `p2p` were ever trainable. This was long-standing, not a
regression: it reproduces identically at `5d7e8d9`.

Correcting it is not a mechanical repair. It decides which positions the orthographic loss
scores, and therefore what a `p2o` or `op2op` run optimizes. Two one-line candidates existed:

1. **Widen the target** to `enc_input_ids[:, 1:]`, so every decoder position is scored against
   the next token in the encoder sequence.
2. **Narrow the logits** to `logits["orth"][..., :-1]`, keeping the `[:, 2:]` target and
   dropping the decoder's final position.

The character tokenizer lays each sequence out as `enc = [LANG, BOS, ...chars, EOS, PAD...]`
and `dec = [LANG, BOS, ...chars, PAD...]`, so `dec_len == enc_len - 1`.

## The evidence that did not decide it

A loss trajectory was the obvious instrument and it turned out to be useless here. Over 12
AdamW steps at `d_model=32`, batch 8, single-threaded BLAS:

| arm | first | last | drop |
|---|---|---|---|
| `p2o` option 1 | 4.6183 | 3.8515 | 16.6% |
| `p2o` option 2 | 4.7310 | 4.0322 | 14.8% |
| `p2o` **shuffled target** | 4.8693 | 3.9703 | **18.5%** |

The control arm, whose target rows are randomly permuted so no input/target correspondence
survives, descends *further* than the correct alignment. Twelve steps of descent therefore
measures the marginal character distribution, not the mapping, and says nothing about which
alignment is right. Any future argument from a short loss trajectory should be discarded on
the same grounds.

Script: `descent_225.py`, run at `825293e`, CPU, torch 2.12.0+cu130, seed 5,
`OMP_NUM_THREADS=1`.

## The evidence that did decide it

Generation is the consumer of whatever the loss trains, so it is the oracle. The orthographic
decoder loop (`model.py:760`) seeds a `(batch, 1)` tensor holding `[BOS]` alone, and the first
token it samples is the first character of the word.

Read against the training layout for the word "long", the two options pair the decoder input
with a different target:

| decoder input | option 1 target | option 2 target |
|---|---|---|
| `--` | `[BOS]` | `l` |
| `[BOS]` | **`l`** | **`o`** |
| `l` | `o` | `n` |
| `o` | `n` | `g` |

Option 2 trains `[BOS]` to emit the *second* character. Generation asks it for the first.

That prediction was then confirmed behaviourally. Both arms were trained to memorise a closed
eight-word list, 600 full-batch steps at `d_model=128`, then greedily generated from:

| arm | final loss | first character correct | starts at the 2nd character |
|---|---|---|---|
| option 1 | 0.0064 | **8/8** | 0/8 |
| option 2 | 0.0056 | 0/8 | **8/8** |

Both fit their objective to near-zero loss. They fit *different mappings*. Option 2 produced
`'oongl'` for **long** and `'eeelow'` for **yellow**: the word, beginning one character late.

Script: `generate_225b.py`, same conditions.

## Decision

The orthographic loss and the orthographic metrics both target
`orthography.enc_input_ids[:, 1:]`.

`compute_loss` asserts the target width equals the logits width and raises a `ValueError`
naming both shapes, rather than leaving the mismatch to `CrossEntropyLoss` and its
`Expected target size [8, 8], got [8, 7]`.

`calculate_orth_metrics` makes the same change. A metric scoring different positions than the
loss trains would report on a model that was never optimized.

## Consequences

`p2o` and `op2op` become trainable for the first time. No historical `p2o` or `op2op` run
exists to invalidate, because none could ever have completed a step.

Orthographic metrics now cover one more position per sequence, the one where the decoder is
asked to emit `[BOS]` from `[LANG]`. That position is trivially predictable and slightly
inflates letter-wise accuracy relative to a metric over content characters only. It is kept
because loss and metric agreeing matters more, and splitting them is what produced this class
of defect in the first place.

A separate mismatch was found while measuring this one and is **not** addressed here:
generation seeds `[BOS]` at position 0, while training places it at decoder position 1 because
`[LANG]` precedes it, so `[BOS]` receives a different position embedding in the two regimes.
It is visible in the option-1 output as a trailing artifact (`'longl'`, `'pencilp'`,
`'yellowy'`: the word followed by a spurious repeat of its first character). Both loss options
share it. Tracked separately.
