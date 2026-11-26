# 0017. Positions are computed for any length

Status: Accepted
Date: 2026-11-24
Supersedes parts of [0002](0002-phoneme-row-ids-replace-ragged-feature-lists.md) and
[0008](0008-generation-seeds-the-training-prefix.md), and amends
[0011](0011-one-table-says-what-each-pathway-reads-and-writes.md),
[0014](0014-a-config-that-validates-runs-every-pathway.md),
[0015](0015-language-belongs-to-each-word.md) and
[0016](0016-a-training-example-is-an-input-and-a-target.md)

## Context

Position was two learned tables of 30 rows, one for each modality:
`orth_position_embedding` and `phon_position_embedding` in
`bridge/domain/model/model.py`. `Model._positions` added a table's rows to the token
embeddings. Think of a street with 30 numbered houses: a token was told its house number,
and there was no house 31.

Four things followed from the tables, each observed at `a3cb285`, the commit before this
change.

| what the tables implied | how it showed |
|---|---|
| A 31st token had no row | A 31-token input raised on 25 of 25 operations, and a 31-token target on the 10 that train or evaluate (`pr3/confirm/accept/lengths.py`) |
| Generation was bounded by the table | `max_orth_seq_len` and `max_phon_seq_len`, both 30, set the step budget of the two generation loops (0008) |
| A slot never trained was noise | Trained on phrases of up to 19 tokens, a model reading 23 to 30 tokens scored 0.000 [0.000, 0.003] (`pr3/confirm/e2e/compare.out`) |
| Mixed-language phrases did not fit | 146 of 208 sample phrases passed 30 tokens (`docs/specs/input-target-pairs.md`, section 1) |

Six schemes were compared on one harness at `9e7d9f2`, on three tasks, six seeds each
(`res_positions.md`, `wf_positions/final/table_pooled_6seeds.md`). At trained lengths
they tied. They split on one cell: hear a phrase of 23 to 30 tokens and spell its last
word, having trained on at most 19.

| scheme | reads the longer input | step time |
|---|---|---|
| the 30-row table | 0.000 [0.000, 0.003] | 1.00 |
| a wider table | 0.000 [0.000, 0.003] | 1.00 |
| sinusoidal | 0.002 [0.000, 0.007] | 1.00 |
| word and letter counters | 0.365 [0.307, 0.430] | 1.01 |
| rotary, rotary mixer | 0.487 [0.277, 0.940] | 1.05 |
| **rotary, stock mixer** | **0.858 [0.357, 0.990]** | 1.03 |

## Decision

**Position is a rotation inside self-attention, computed for any length.** This is
rotary position embedding. Each query and each key is turned by an angle proportional to
its position before the dot product, so a score depends on how far apart two tokens are
and not on which slots they fill. Think of the two hands of a clock: move both on by an
hour and the angle between them is what it was.

These are decisions 7 to 10 of `docs/specs/input-target-pairs.md`.

| point | rule |
|---|---|
| Where it applies | The self-attention of the two encoders and of the two decoders |
| Where it does not | The mixer (the stack that writes the global memory) is torch's stock `nn.TransformerEncoder` and gets no position. A decoder's cross-attention to the global memory gets none, and neither do the two cross-attentions between modalities on `op2op` |
| One scheme | No configuration switch, and no second scheme kept beside it |
| The one function | `rotate` in `bridge/domain/model/layers.py`. Pair `i` of a head's `n` pairs of dimensions turns by `position * 10000 ** (-i / n)` |
| Owned blocks | `layers.py` holds `Attention`, `EncoderLayer`, `DecoderLayer`, `Encoder` and `Decoder`. Torch's stock layers give no access between the query and key projections and the dot product |
| Parameter names | Those of torch's own stacks, so a stock stack's `state_dict` loads into an owned one |
| Embeddings | A token embedding alone. The tables, `Model._positions` and every `position_offset` are removed |
| Length | Nothing checks it. `max_orth_seq_len`, `max_phon_seq_len` and the length check at the generation boundary are removed |
| Generation | `Model.generate(inputs, pathway, *, max_new_tokens, orth_start=None, deterministic=False)` |
| Head size | `ModelConfig` refuses a `d_model` that `nhead` divides into heads of one dimension |
| `d_embedding` | Defaults to 2, where it defaulted to 1 |
| Checkpoints | One from a model with position tables is refused. There is no migration |
| `VocabSpec.orth_language_ids` | Required. `TrainingPipeline` reads it with no fallback, and the `_without_switches` accuracy pair is always reported |

**Why the blocks are owned.** The rotation sits between the projections and the dot
product. The nearest thing torch's layers accept is a float attention mask added to the
scores, and they do not carry one faithfully. The decoder's generation call drops it
(deviation 0.000e+00 from the unbiased output). The encoder's inference fast path reads
it as a boolean, and with padding returns 960 non-finite values of 2496
(`wf_positions/probes_rope/trap_stock_layers.out`). So the blocks are owned, and held to
torch's by a reference test: with the rotation off they compute what the stock stacks
compute on the same weights.

**Why the mixer is left alone.** With a rotary mixer the headline cell scored 0.487
against 0.858. The mixer writes the global memory, whose rows have no order, from tokens
the encoders have already placed.

## The length cap

A table of 30 rows was also the reason generation stopped. Without it nothing in the
model says when to stop, so the caller does.

`max_new_tokens` is keyword-only and required, a positive `int` with no default. It
counts the tokens written after the start, which is `[LANG, BOS]` for orthography and
`[BOS]` for phonology. A sequence stops at `[EOS]` or at the cap, whichever comes first.
One that reaches the cap ends without `[EOS]`, which is how a caller sees it was cut.
0009 stands: a finished sequence emits padding. On `op2op` the one cap applies to each
modality written.

What the main generation routines do, read from their documentation on 2026-11-22:

| framework | the cap | when the caller omits it |
|---|---|---|
| Hugging Face `generate` | `max_new_tokens`, new tokens only | 20 tokens in total, with a warning. The docs say to always set it |
| fairseq | `max_len_a * source_length + max_len_b` | 0 and 200, so a flat 200 |
| llama.cpp | `n_predict` | -1, no limit |
| vLLM chat endpoint | `max_tokens` | whatever is left of the model's context window (from a secondary source) |
| Anthropic Messages API | `max_tokens` | rejected; the field is required |

All five have a cap and stop at end-of-sequence. They split on omission. The defaults
that work are backed by a fixed context window, which this model does not have. The one
that is not, Hugging Face's 20, cuts output in silence, and its own docs warn against
relying on it. So the name and the counting follow Hugging Face, and the missing default
follows Anthropic.

## The `d_embedding` default

`d_embedding` is the number of rows of the global memory, which is all a decoder reads of
the input. It defaults to 2. The change lands here because it moves every default
model's output, and this change already does.

The research measured 1, 8 and 32 rows with the sinusoidal scheme only, and nothing about
2 (`wf_positions/audit/trainlong/table_trainlong.md`). The shipped code was then measured
at 1, 2, 8 and 32. The task is `o2o` repetition with `d_model` 64, 3500 steps, scored by
greedy generation on 300 held-out phrases. Cells are the median with [min, max]
(`pr3/confirm/e2e/cell.py`, `sweep.py`, `summary.out`).

Trained on phrases of 5 to 47 tokens. Three seeds at 1, 2 and 8 rows, two at 32.

| rows | 9 to 19 tokens | 23 to 30 | 33 to 47 | 51 to 61, past training |
|---|---|---|---|---|
| 1 | 0.460 [0.450, 0.473] | 0.017 [0.017, 0.023] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 2 | 0.817 [0.810, 0.830] | 0.223 [0.217, 0.237] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| 8 | 0.987 [0.983, 0.993] | 0.977 [0.977, 0.983] | 0.620 [0.610, 0.690] | 0.000 [0.000, 0.000] |
| 32 | 0.993 [0.987, 1.000] | 0.988 [0.983, 0.993] | 0.895 [0.870, 0.920] | 0.003 [0.003, 0.003] |

Trained on phrases of 5 to 19 tokens. Three seeds at 1 and 2 rows, six at 8.

| rows | 9 to 19 tokens | 23 to 30, past training |
|---|---|---|
| 1 | 0.593 [0.567, 0.610] | 0.000 [0.000, 0.000] |
| 2 | 0.867 [0.847, 0.890] | 0.000 [0.000, 0.000] |
| 8 | 0.985 [0.957, 1.000] | 0.000 [0.000, 0.003] |

So 2 rows are better than 1 at every length, and they do not carry a phrase. With 2 rows
a model repeats 0.867 of short held-out phrases and none of 33 tokens or more. The
default suits single words. A run on phrases sets `d_embedding` itself.

## What was measured

Measured on 2026-11-24, CPU, float32, one thread, torch 2.12.0+cu130, python 3.12.13,
`BRIDGE_DEVICE` unset. The working tree is `a3cb285` plus this change, and the baseline
is a pristine worktree of `a3cb285`. Integer ids and masks are compared with
`torch.equal` plus dtype and shape. Floats are compared by the largest difference over
the largest magnitude of the reference, against a floor measured in the same run.

**Position is the only thing that moved.**

| check | result | script |
|---|---|---|
| Everything that does not pass through the weights, against `a3cb285`: tokenizer tensors over 3000 English words, 3000 Spanish words, 600 phrases and 600 mixed rows, every batch of four csv files, and the form of every forward and generation output | 0 differences over 21462 entries, 11428793 elements. Each tree against itself: 0. 5 of 5 plants caught | `pr3/confirm/accept/exact.py` |
| The baseline with its tables zeroed against the working tree with the rotation off, the same weights loaded into both. Five pathways, two seeds, eval and train | Logits bitwise equal, 551424 elements. 44 loss terms bitwise equal. 2068 gradient tensors within 6.955e-07, which is 0.91 of the floor between one thread and four. 5 of 5 plants caught | `pr3/confirm/accept/unchanged_but_position.py` |
| The same comparison by another script, at `d_embedding` 1 and 2 | Logits within 2.560e-07 to 3.649e-07 on five pathways, every generated id equal. Controls 6.927e-01 to 1.129e+00 | `pr3/confirm/rc/p9_whole_model.py` |

**The five correctness probes of the bake-off pass on the shipped code**, each with its
planted defect caught: 55 of 55 honest checks and 26 of 26 plants
(`pr3/confirm/accept/correctness.py`).

| probe | claim | honest reading | planted defect reads |
|---|---|---|---|
| A | An owned stack with the rotation off is torch's stock stack on the same weights | 0.000e+00, every element bitwise, against the stock training path. 2.944e-07 to 6.331e-07 against the stock inference path, which is what the stock stack's own two paths differ by | Rotation on: 7.514e-03 to 1.855e-01 |
| B | Moving every position by `k` changes nothing | 2.620e-07 to 8.271e-07 at `k` of 1 and 17; 1.028e-05 at most at 1000 | Half the positions moved by 3: 7.916e-02 to 1.708e-01 |
| C | Eval and train agree once dropout is off | 3.709e-07 and 4.260e-07, against 3.671e-07 for the baseline on the same measure | 1.905e-01 and 2.186e-01 |
| D | Generating step by step equals one full decode of the result | 2.384e-07 to 5.364e-07. 0 of 4628 decisions differ | 6.460e-01 to 7.833e-01 |
| E | A padded batch scores and generates each row as the row alone | Logits 4.133e-07. 0 of 108 generated sequences differ | 3.079e-01 and 3.783e-01; 88 of 108 sequences differ |

Probes C, D and E ran on rows of up to 30 tokens and again on rows of up to 80.

**Any length runs.** Rows of 31, 60 and 200 tokens train, evaluate and generate on five
pathways: 75 of 75 cells ran. The baseline ran 25 of 25 at 30 tokens and raised on 25 of
25 with a 31-token input (`pr3/confirm/accept/lengths.py`). A file whose longest row is
2021 orthographic tokens and 1858 phonological ones steps, evaluates and generates on
all five pathways, with every loss and gradient finite. From that input, with `[EOS]`
excluded, `op2op` at a cap of 150 writes exactly 150 in each modality
(`pr3/docs/long/p6_long.py`, `p6_long.out`).

**The cap does what it says.** 32 of 32 checks, exact
(`pr3/confirm/accept/cap.py`). With `[EOS]` made impossible, a cap of `k` writes exactly
`k` tokens. A trained model wrote 2304 rows over 120 runs: 1502 finished and 802 cut. No
row passed the cap. Every finished row holds one `[EOS]` and only padding after it. Every
row with no `[EOS]` ran to the cap. Three planted loops were each reported: one step past
the cap, no padding after `[EOS]`, and a cap that counts the start.

**A value that is not a positive `int` is refused.** 100 of 100: ten bad values on five
pathways, by `generate` and by `_generate`. The message names the value and its type, as
in `got 5 (int64)` and `got True (bool)`. 35 of 35 legitimate caps are accepted,
`int(numpy.int64(7))` among them (`pr3/confirm/probes/cap_uses.py`).

**The sizing rule in `generate`'s docstring is exact.** Orthography of `n` characters
takes `n + 1` tokens and one more for each change of language: equal to the labels' count
on 7200 of 7200 rows. Phonology takes one for each phoneme, one for each gap between
words and one for `[EOS]`: equal on 7200 of 7200, with the phonemes counted from the
lexicon file. A decoder scripted to write the gold target holds `[EOS]` in every row at
a cap equal to the rule, and cuts only the longest row at one less (`cap_uses.py`).

**The shipped code reproduces the headline cell.** The task hears a phrase and spells
its last word, with `d_model` 64 and `d_embedding` 8, trained on 5 to 19 tokens for 3500
steps, six seeds, greedy exact match (`pr3/confirm/e2e/compare.out`).

| tree | 9 to 19 tokens | 23 to 30 | 33 to 47 |
|---|---|---|---|
| working tree | 1.000 [1.000, 1.000] | 0.848 [0.450, 1.000] | 0.130 [0.010, 0.753] |
| `a3cb285` | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.003] | raises on 6 of 6 seeds |

The bake-off's cell was 0.858 [0.357, 0.990]. The per-seed scores are 0.450, 0.813,
1.000, 0.490, 0.883 and 0.997, so the spread across seeds is most of the range.

These 35 cells, 29 on the working tree and 6 on the baseline, were run twice: before the
last round of review fixes, and after it. All 35 are equal in every compared value. The
comparator reports two seeds as differing in 12 of 23 values, and reports one value
moved by one part in 4e15 (`pr3/confirm/e2e/compare.py`). The scorer gives the gold
target 1.0 and each of three plants 0 on 169 of 169 scored splits.

**Further end-to-end runs, on the tree before that last round of fixes.** The round
changed four library files, and the 29 cells above show it moved no number they hold.
These were not run again (`pr3/e2e/summary.out`).

| run | result |
|---|---|
| Trained on up to 47 tokens, spelling the last word of a 51 to 61 token phrase, three seeds | 0.917 [0.890, 1.000]. The bake-off's cell was 0.932 [0.887, 0.977] over two seeds |
| 8 rows of 60 tokens, `d_embedding` 32, 1500 steps | Exact match 1.000 on `o2p`, `p2o` and `p2p`, with 0 rows cut |
| Mixed-language targets, 300 held-out rows, three seeds | `o2o` 1.000 [1.000, 1.000] with 225 of 225 switch tokens right on each seed. `p2o` 0.997 [0.993, 1.000] |
| Next word, 300 held-out phrases, three seeds | `o2o` and `p2o` 1.000 [1.000, 1.000]. With targets permuted, 0.023 and 0.017 |
| Repetition at trained lengths against `a3cb285`, same data, six seeds | `o2o` 0.985 [0.957, 1.000] against 0.992 [0.990, 1.000]. `o2p` 0.925 [0.890, 0.967] against 0.950 [0.913, 0.970] |

The last row is tied on its stated test and leans against this change. On `o2p` the
working tree is below the baseline on 4 of 6 seeds, with a paired mean of -0.021 and a
spread of 0.046 across seeds. The bake-off leaned the other way on the same task, 0.960
against 0.942. Six seeds do not settle it.

**The noise floor of this model.** 12 steps of `single_step`, `d_model` 32, `nhead` 2,
default `d_embedding`, seed 0, batch 8, over `o2p`, `p2p`, `p2o` and `op2op`. This
machine's default is 6 threads (`pr3/confirm/floor/floor_side.py`, `floor_compare.py`,
`floor.out`).

| tree | comparison | max relative loss deviation | relative L2 parameter distance |
|---|---|---|---|
| working tree | rerun at one thread | 0.000e+00 | 0.000e+00, 167 of 167 tensors bitwise |
| working tree | one thread against 6 | 1.850e-07 | 6.223e-06 |
| working tree | one thread against 4 | 1.480e-07 | 7.279e-06 |
| `a3cb285` | rerun at one thread | 0.000e+00 | 0.000e+00 |
| `a3cb285` | one thread against 6 | 1.327e-07 | 5.856e-06 |
| `a3cb285` | one thread against 4 | 1.327e-07 | 6.769e-06 |

`global_embedding` scaled by 1 + 1e-3 reads 1.853e-04 and 7.063e-04, with 0 of 12 losses
bitwise on every pathway. Scaled by 1 + 1e-6 it reads 1.850e-07 and 7.958e-06, which is
inside the floor. A change that small cannot be told from a thread count.

## Consequences

**Removed.** The two tables, `Model._positions`, `max_orth_seq_len`, `max_phon_seq_len`,
the length check in `_validate_encoder_inputs`, every `position_offset` parameter, and
`Model.generate_triangular_mask`: a `Decoder` builds its own causal mask. A scan for 14
patterns of removed names finds 42 hits in `bridge/` at `a3cb285` and none in the working
tree (`pr3/confirm/scan/scan.sh`, `scan.out`).

**The model has 167 parameter tensors, where it had 169.** A default model holds 625779
scalars at `d_embedding` 2. At `a3cb285` it held 629555 at `d_embedding` 1. The two
tables were 3840 of them, and the second memory row adds 64
(`pr3/confirm/param_count.py`, `pr3/confirm/accept/parameters.py`).

**Parameter names follow torch's stacks.** 144 of the 167 tensors dropped a name segment
the wrappers of `a3cb285` added, `transformer_encoder` or `transformer_decoder`. With
that segment dropped, 167 of 167 names and shapes match, in the same order. The
acceptance probe `parameters.py` asks for the old names verbatim and prints
`FAIL checks 32/33` on that one check. The names are what lets a stock stack's
`state_dict` load into an owned stack, which probe A depends on, and no checkpoint with
the old names loads either way.

**`layers.py` is 188 lines, where it was 66**, 114 of them code against 44. `bridge/` is
4124 lines against 3991.

**A training step costs about 3% more.** Each round steps five processes in turn on
one pinned core: the baseline, the working tree, each of them a second time, and the
baseline with a 2 ms busy-wait planted in its step. 150 timed rounds after 20 discarded,
`d_model` 64, `d_embedding` 8, load average 2.4 to 3.0. Cells are the median over rounds
of the paired difference in a whole `single_step`
(`pr3/docs/x5/timing_coord.py`, `timing_worker.py`, `timing_final.out`).

| pathway, batch | working tree against `a3cb285` | the same code twice: baseline, working tree | the 2 ms plant reads |
|---|---|---|---|
| `o2p`, 32 | +4.11% | -0.29%, -1.31% | 2.07 ms |
| `op2op`, 32 | +2.88% | -0.11%, -0.16% | 1.76 ms |
| `o2p`, 128 | +1.80% | +0.20%, -0.71% | 1.88 ms |
| `op2op`, 128 | +1.19% | -1.47%, -0.24% | 2.05 ms |

The third column is the floor. The last row is not a result: its floor, 1.47%, is larger
than its effect. Timing the forward, backward and optimizer step alone gives +3.90%,
+2.98%, +2.26% and +1.59%, and reads the plant as 1.26 ms in the last row. The same
harness on the tree before the last round of review fixes read +2.6%, +3.1%, +2.4% and
+1.8% (`pr3/e2e/x5/timing_run2.out`). The bake-off measured 3.4% for this scheme.

**Checkpoints saved at `a3cb285` or before do not load, and there is no migration.**
This is decision 9 of the spec. `load_model`, and a pipeline built with
`TrainingConfig.checkpoint_path`, compare every tensor name and shape with the model
before copying anything. They raise `RuntimeError` naming the file, the count of tensors
that differ and the first of each kind. Six wrong checkpoints are refused with 0 of 167
tensors changed, and the optimizer and `start_epoch` unchanged: a real `a3cb285` file,
the same file with its table fingerprint deleted, either direction of a `d_embedding`
mismatch, another width, and another layer count. Eight legitimate loads are accepted
(`pr3/confirm/probes/ckpt_uses.py`, `p5_checkpoints.py`).

The check exists because torch's own refusal comes late. `model.load_state_dict` on an
`a3cb285` checkpoint raises only after overwriting 21 of 167 tensors whose names happen
to match. The library never calls it without the check, and never passes `strict=False`.

This supersedes one sentence of 0002, that existing checkpoints keep loading under
`strict=True`. The buffer that sentence is about is still non-persistent.

**The golden master is not regenerated.** `tests/fixtures/phon_baseline.pt` (the
recorded fingerprint of the phonological path) is byte-identical to its blob at
`a3cb285`, sha256 `6fcb20933ec3f3a4`, and so is its capture script. Its 24 test nodes
pass (`pr3/confirm/accept/static.py`).

The recording was made by a model with the two tables. Without them a seed's draws fall
differently across the parameters that remain, so no recorded embedding, logit, encoding
or generated token can be produced again. The spec expected 12 nodes to stay as they
were and 12 to be reduced. Measured by mutating a copy of the recording 255 ways and
running both trees' nodes on each, the split is 11 and 13.

| nodes | how they read the recording |
|---|---|
| 9 tokenizer nodes and 1 vocabulary node | Exactly, as at `a3cb285` |
| 1 gradient node | At the tolerance it has always had. The gradient of an embedded batch with respect to the phonological embedding counts features, and no weight enters it |
| 1 position-gradient node | Exactly, where it had a tolerance. The recording is checked to be the count of rows in each slot, and the phonological embedding is checked to be the only parameter an embedding reaches |
| 2 embedding nodes, 4 logit nodes, 2 generation nodes that write orthography alone | Reduced to form: which outputs are filled, and the shape, dtype, finiteness and id range of each |
| 3 generation nodes that write phonology | Reduced to form, plus the recorded `[BOS]` that opens each sequence and the recorded length of each |
| 1 node for an input without phonology | Still exact on the recorded stand-in. Its generation is compared with an input carrying both modalities, where it was compared with the recording |

No tokenizer or vocabulary node is weakened and none is deleted. No node was emptied:
each still fails on some change to the recording. New behaviour is pinned by reference
and analytic oracles in `tests/domain/model/test_rotary_layers.py`,
`tests/domain/model/test_any_length.py` and `tests/domain/model/test_max_new_tokens.py`,
not by a re-recording.

**The tests can see the defects they are aimed at.** Each plant is one defect in a copy
of the tree, followed by the whole suite.

| plants | result | script |
|---|---|---|
| 30, one for each gap the reviews found | 28 of 28 defects fail every test aimed at them. 2 controls fail nothing: a comment, and a refactor that turns queries and keys in two calls | `pr3/confirm/mutate/mutate2.py`, `results2.jsonl` |
| 15 of the confirming run's own | 14 fail a test, and the control fails nothing | `pr3/confirm/mutate_own/mutate_own.py`, `results_own.jsonl` |

The one survivor applies the decoder's pad mask to queries instead of keys. It is
equivalent on every batch the tokenizer can build, since padding sits to the right under
a causal mask: logits at real positions differ by 0.000e+00 against a thread floor of
1.984e-07. A hand-built mask with padding before a real token moves them by 2.303e-01
(`pr3/confirm/probes/n10_equivalent.py`).

**The suite** is 1042 passed and 6 skipped, against 873 and 6 at `a3cb285`. Over three
runs it takes 11.28 to 11.72 s by pytest's own count, against 10.23 to 10.34 s for
`a3cb285` with the two run alternately, load average 2.5 to 3.1
(`pr3/confirm/time_suites.sh`, `timing.out`). With the thread variables unset it passes
the same 1042, so no new tolerance is tuned to one thread
(`pr3/confirm/suite_default_threads.out`). Those counts are of the library and its tests.
With the three documentation tests below and one test of the figure, the tree was 1046
passed and 6 skipped in 11.39 to 11.55 s, against 10.07 to 10.22 s for `a3cb285` in the
same session, load average 1.4 to 2.3 (`pr3/docs/suite_time.sh`, `suite_time.out`).
`.claude/rules/measurement.md` carries the current count.

**The documents are held by tests and by a probe.** `tests/test_documentation.py` gains
three checks: the call the map shows for `Model.generate` against the method's signature,
the two tables of sizes against the defaults of `ModelConfig`, and the recorded parameter
count against a default model. 12 of 12 planted document defects fail their guard, 9 on
the new checks and 3 on the check that runs the README's examples
(`pr3/docs/doc_test_plants.py`). Beyond the suite, 77 sentences of the README and the map
were each checked against the running code or against a recorded output, 82 checks in
all, with 3 of 3 planted defects failing: two false sentences and one sentence the map
does not hold (`pr3/docs/doc_claims.py`). Every example in the README runs as written
with CUDA visible and with it hidden (`pr3/docs/readme_blocks.py`).

**On a GPU the model behaves as it does on CPU.** This is placement and control flow, on
an RTX 5080, and no CUDA float is compared with a CPU float. A model built on the process
device sits on `cuda:0`. All five pathways generate with no explicit `.to()`. One
optimizer step and one `evaluate` give a finite loss on each pathway. A batch holding a
102-token row steps, evaluates and generates. A seed gives the same parameters on both
devices, 167 of 167 tensors bitwise (`pr3/docs/cuda_behaviour.py`).

## Limits

Two hold for every scheme the bake-off ran, and the shipped code shows both.

- **Nothing repeats a phrase longer than it trained on.** Exact match is 0.000 to 0.003
  in every cell past the trained lengths, in both tables of the `d_embedding` section.
  The bake-off measured 0.000 for all six schemes. Reading a longer input is a different
  task and does extend, as the headline cell shows. Training data must cover the lengths
  in use.
- **A small global memory cannot carry a phrase.** One row repeats none of 33 to 47
  trained tokens, and neither do 2.

Seven more belong to this change.

- **The angle is held in float32, whatever the model's dtype.** Moving every position by
  the same amount should change nothing. It changes the output by 7.324e-06 at a move of
  1000, 1.490e-04 at 1e5 and 2.875e-02 at 1e7. With a float64 angle the same moves read
  4.9e-07, 4.3e-07 and 7.1e-07 (`pr3/confirm/accept/angle64.py`). It is harmless at the
  lengths this model reads. A float64 angle was declined because torch documents that
  the MPS backend has no float64, which was read and not measured here.
- **`d_model=2` with `nhead=1` validates and cannot read order.** "cat" against "act"
  differs by 0 to 1.0e-07 over three seeds, and did at `a3cb285` too. Every refused size
  from `d_model` 4 to 64 is as blind, 4.8e-08 to 3.1e-07, and every accepted size from
  `d_model` 3 up reads order, 3.8e-04 to 1.2e-01. `ModelConfig` agrees with "`nhead`
  divides `d_model` into heads of two or more" on 583 of 583 sizes
  (`pr3/confirm/probes/headdim.py`). A floor of 3 on `d_model` would close it.
- **A phonological row counts as finished when the `[EOS]` feature is on beside other
  features.** At a cap of 40, 2 of 64 rows of a trained model end that way, on features
  (29, 32) and (32, 34). A caller testing for a position equal to `[EOS]` counts them as
  cut (`pr3/confirm/capprobes/cap_probe2.py`). `generate`'s docstring states the rule.
  Whether the loop should demand a clean `[EOS]` is undecided.
- **Nothing bounds `max_new_tokens`, and a row that never stops runs the batch to the
  cap.** Each step decodes the whole prefix. An untrained `o2o` model of 64 rows took
  0.93, 2.56, 10.13 and 43.17 s at caps of 40, 80, 160 and 320, on a machine at load 19
  to 20, so read the growth: about four times per doubling. A trained model took 0.42 s
  at caps of 40 to 160 (`pr3/confirm/capprobes/cap_probe.py`).
- **A cut row and a finished shorter row can decode to one string.** `decode` drops
  `[EOS]`, so the test is on the ids (`pr3/confirm/capprobes/decode_cut.py`).
- **An all-padding row gives finite output, where torch's stock stack gives NaN.** In
  eval, such a row gives 352 NaN from each stock stack and 0 from each owned stack
  (`pr3/confirm/rc/p1_layers.py`). The tokenizer builds no such row.
- **Dropout draws differ from torch's by layout, not by count.** With attention dropout
  alone, an owned stack equals the stock one at every size. With residual and
  feedforward dropout, the same number of draws falls on other elements at a batch above
  one, 9.4e-02 to 2.1e-01 (`pr3/confirm/accept/dbg_dropout.py`). The reference test
  lays the draws out as torch does.

Issue #233 stands as 0016 left it. A half-precision model generates on `o2p` and `p2p`,
and is rejected with `orth_probs probabilities must sum to 1` on the three pathways that
write orthography, in float16 and in bfloat16 (`pr3/docs/issue_233.py`).

Three spots of `model.py` are not reached by the suite, as at `a3cb285`: the error for a
phoneme vocabulary of the wrong size, the sampling branch of `ortho_sample`, and the
orthographic loop's return for an empty batch (`pr3/confirm/cov/covplug.py`).

Not measured:

- CUDA numerics. Every number in this record but the GPU paragraph above is CPU, and no
  noise floor exists for CUDA.
- `d_model` above 64, more than 5000 steps, real corpora, or new words.
- `p2p` and `op2op` scored on a trained task. They ran in the correctness, cap and
  length probes only.
- The switch token in a row longer than the lengths trained on.

Scripts and outputs are under `pr3/` in the evidence folder that
`docs/specs/input-target-pairs.md` names. `pr3/confirm/` holds the confirming run of
every probe on the final tree, whose library files have the digest `d8ac7bfa95c23678`
(`pr3/confirm/bridge_files.sha256`). `pr3/e2e/` holds runs on the tree before the last
round of review fixes. `pr3/docs/` holds the runs made on the final tree while the
documents were written.
