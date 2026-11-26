# 0016. A training example is an input and a target

Status: Accepted, amended by [0017](0017-positions-are-computed-for-any-length.md)

> The five types and the two sides still hold. One detail is amended. `Model.generate`
> also takes `max_new_tokens`, keyword-only and required, the most tokens to write after
> the start of each sequence. See 0017.
Date: 2026-11-23
Amends [0004](0004-orthographic-teacher-forcing-alignment.md) and
[0013](0013-the-caller-owns-the-loop-in-fact.md), and details of
[0008](0008-generation-seeds-the-training-prefix.md) and
[0015](0015-language-belongs-to-each-word.md)

## Context

Every training example was one string, which the model read and then reproduced.
`EncodingComponent` (the five tensors of one modality of one string) held both what an
encoder reads and what a decoder is taught: `enc_input_ids`, `enc_pad_mask`,
`dec_input_ids`, `dec_pad_mask` and `targets`. `BridgeEncoding` paired one of them per
modality. That only makes sense when the two are the same string. Think of a flashcard
with the same text printed on both sides.

Four things followed from the one type, each observed at `7386f8e`.

| what the type implied | how it showed |
|---|---|
| The orthographic loss scored the encoder's own ids | The labels were `enc_input_ids[:, 1:]`, sliced once in the loss and again in the metrics (0004) |
| Generation took its opening language from the input | The seed was `dec_input_ids[:, :2]`, taken off the encoding (0008) |
| A missing modality was a placeholder tensor | A phonology-only encoding carried `[LANG, BOS]` as its orthography. Three pathways read it and died inside the encoder (issue #233) |
| A data file had no place for a target | A row was `word_raw` and `language` |

`Model.forward` needed no change. It takes tensors, and a decoder reads the global memory,
whose size does not depend on the input's length.

Three designs were prototyped on `9e7d9f2` and judged (`res_design.md`). The cheapest kept
the two types and passed two `BridgeEncoding` objects per example. Nothing in it says
which half is which. With the halves exchanged, forward and loss ran and returned 4.742.
A file whose target column was misspelt loaded and trained repetition
(`wf_design_judge/hunt.out`).

## Decision

**An example is an input and a target, two different things by design, each with its own
types.** The model reads the input and is taught to write the target. They may be equal:
repetition is a dataset whose two columns match, and no code treats it differently.

These are decisions 1 and 2 of `docs/specs/input-target-pairs.md`: five types named for
their side, and a required `target_raw` column.

```python
class Tokens:        ids, pad_mask                 # what an encoder reads
class Target:        ids, pad_mask, labels         # what a decoder is fed, and scored against
class BridgeInput:   orth: Tokens | None, phon: Tokens | None
class BridgeTarget:  orth: Target, phon: Target
class BridgeBatch:   input: BridgeInput, target: BridgeTarget
```

`EncodingComponent` and `BridgeEncoding` are deleted, not kept beside them.

| point | rule |
|---|---|
| Names | The modalities are `orth` and `phon`, the words `PATHWAY_IO` (the table of what each pathway reads and writes) uses |
| One reader | Every tensor has exactly one reader. An encoder reads `Tokens`. A decoder is fed a `Target`'s `ids` and `pad_mask`. The loss and the metrics score against its `labels` |
| Validation | Each type is frozen and checks itself when built. Ids and mask agree in shape. Labels are int64 and give one row per decoder position. Batch sizes and devices agree |
| Roles | An input where a target belongs raises. `BridgeBatch(input, input)` raises `TypeError`, and so does `Model.generate` handed a `BridgeTarget` |
| Orthographic labels | Stored on the target, like the phonological ones. Ids are `[LANG, BOS, chars, PAD]` and labels are `[BOS, chars, EOS, PAD]`. A mid-row language token sits in both |
| A missing modality | `None`, not a placeholder tensor. A pathway asked to read it raises `ValueError` naming the pathway and the modality. `BridgeInput` takes each modality by name |
| Targets | A target carries both modalities, always |
| Tokenizer | `encode_input(texts, languages)`, `encode_target(texts, languages)` and `orth_start(languages)`. The old `encode` is removed |
| Dataset columns | `word_raw`, `language`, `target_raw`, `target_language`. `target_raw` is required, and the error names the fix: for repetition, copy `word_raw` |
| A missing `target_language` | Copies `language` when that is one code. A per-word `language` raises at load, naming the row |
| Dataset storage | One list of `Example` rows. Each holds both strings and both tuples of codes, so a shuffle cannot separate them |
| Reading a row | `dataset[i]` takes an `int` or a `slice` and returns a `BridgeBatch` |
| Pipeline | `forward(batch)`, `compute_loss(logits, target)`, `compute_metrics(logits, target)`. `single_step` and `evaluate` keep their signatures |
| Events | A `train` event's metrics carry `input` and `target`, each a JSON list of the strings the step covered |
| Generation | `Model.generate(inputs, pathway, *, orth_start=None, deterministic=False)` reads the input and nothing else |
| The output language | A pathway that writes orthography requires `orth_start`, the `(batch, 2)` `[LANG, BOS]` rows from `tokenizer.orth_start(["ES", ...])`. Omitting it raises. It names the language of the first word written; later switches come from the decoder |
| Public API | `bridge.__all__` exports the five types |

Four choices the spec left to the implementation:

- **`languages` has one form.** It is a list or tuple with one entry per row, or `None`. An
  entry is one code, a string of codes, a list or tuple of codes, or `None`. `texts` is a
  list. A bare string for either raises, so `["ES", "EN"]` never means two things.
- **A word with no pronunciation is a data condition, not an argument error.**
  `encode_input` returns the input with `phon=None`, and `encode_target` returns `None`.
  There is no `modality_filter`. The research graft took one
  (`wf_design_judge/graft_optional.py`), and returned `None` for a nonword unless the
  caller asked for orthography alone. Here a nonword is always read through its spelling.
  Phonemes without a spelling are `BridgeInput(phon=...)`.
- **The encoding memo and the load-time encode of row 0 are deleted.** The memo ran 0
  times in 6 training steps (`res_design.md`). At `7386f8e` a first row with no
  pronunciation was left out with a warning, and any other such row raised when read.
  Every row behaves the same way: it loads, and it raises when it is read.
- **The dataset refuses a row with no pronunciation on either side, whichever pathway is
  training.** A nonword input reaches a model through `encode_input`, not through a
  dataset.

## What repetition keeps

With the target equal to the input, and `orth_start` naming the input's first-word
language, everything is bitwise equal to `7386f8e` on all five pathways. Measured on
2026-11-23 against a pristine worktree of `7386f8e`, CPU, float32, one thread, torch
2.12.0+cu130, `BRIDGE_DEVICE` unset. Integer ids and masks are compared with `torch.equal`
plus dtype and shape.

| check | result | script |
|---|---|---|
| Tokenizer: 7506 rows as 7749 calls a side. Words, phrases, mixed rows, odd whitespace, capitals, no language, nonwords, wrong-language words, empty rows, batches | 0 of 92268 tensor pairs differ, over 11 old-to-new tensor mappings | `pr2/confirm2/r/accept/diff_tokenizer.py` |
| Training: 200 rows, 12 steps of batch 16, model seeds 0 and 7, five pathways. Logits, losses, metrics, 169 parameters after every step, `evaluate`, greedy and sampled generation | 0 of 225158 leaves differ | `pr2/confirm2/r/accept/diff_training.py` |
| Controls: `7386f8e` against itself, and the working tree against itself | 0 of 225158 each; 0 of 7749 tokenizer calls | both |
| Planted in the training differential: labels shifted one further, one row's language token changed, one `orth_start` row changed, the dataset's sides exchanged, the decoder fed the input's ids | 5 of 5 caught, in the cells predicted | `diff_training.py` |
| Planted in the tokenizer differential: the first three of those, one flipped label, one id tensor cast to int32 | 5 of 5 caught | `diff_tokenizer.py` |
| A second differential by another script: 200 rows, model seeds 5 and 7. Tokenizer tensors, dataset reads, 12 steps with a shuffle between them, `evaluate`, generation | 0 of 75903 values differ. Each tree against itself: 0. With labels shifted one further: 22943 differ | `pr2/docs2/snapshot/snapshot.py`, `compare.py`, `compare_base_vs_new.out` |
| A csv of 200 rows through `BridgeDataset`, 12 steps with metrics, `evaluate` and `generate`, five pathways, seed 0 | 27952 exact values with 0 differing; 8968 of 8968 floats bitwise | `pr2/confirm2/r/e2e/x3_seed0.sh`, `x3_seed0.out` |
| Its controls and plants | Each tree against itself is identical, 10 of 10. One changed target differs on 5 of 5 pathways, and so do labels that never teach `[EOS]`. A wrong `orth_start` differs on the 3 that write orthography and on neither of the others | same |
| Dataset files: 32 a caller may hold, each given a `target_raw` equal to its `word_raw`. Csv, pkl, and frames a storage client filtered, sorted, reversed, de-duplicated or re-indexed. 14 kinds of read each | 32 of 32 load and read as at `7386f8e`. A planted difference is found | `pr2/confirm2/r/s6_regress/legit_vs_base.py` |
| An input carrying orthography alone against the placeholder encoding of `7386f8e`: 80 nonword rows, `o2p` and `o2o`, two seeds | 0 of 3968 leaves differ | `pr2/confirm2/r/accept/diff_nonword.py` |
| An input carrying phonology alone, on `p2o` and `p2p` | 1976 of 1976 values bitwise | `pr2/docs2/snapshot/compare_base_vs_new.out`, row `phononly` |
| The golden master, `tests/fixtures/phon_baseline.pt` (the recorded fingerprint of the phonological path) | File byte-identical to its blob at `7386f8e`; its 24 tests pass | `pr2/confirm2/r/s2_suite/checks.sh` |

The training differential has a second section, where each row's target is another row.
`7386f8e` has no pairs, so its side is wired by hand: encoder tensors from one `encode`
call, decoder tensors and labels from another, through `Model.forward` and its loss and
metrics. The pipeline equals that on every leaf too. The 225158 leaves are both sections.

The shift 0004 defines keeps its values. `target.orth.labels` equals
`enc_input_ids[:, 1:]` on every one of the 7749 calls, which is one of the 11 mappings.

One of the 24 golden master tests asserts less than it did. The fixture records the
placeholder phonology of an orthography-only encoding, and no such tensor exists. The
test asserts that an input carrying orthography alone generates what the fixture records
for `o2p`, following the pattern `.claude/rules/measurement.md` sets for a superseded
value. The nonword rows above are the measurement behind it. The second differential
measures the same thing on three nonword rows: 729 of 729 values, row `nonword`.

Every output names the tree it ran on. The working tree is `7386f8e` plus uncommitted
work, and the sha256 of `git diff HEAD -- bridge` begins `bfae82790321`. A probe that
strips the diff's last newline before hashing prints `5d7a024835ab` for the same tree.
The runs under `e2e/` use a snapshot whose `bridge/` is file for file equal to it.

## What a pair does

Measured on the working tree, same conditions (`pr2/confirm2/r/accept/pair_oracle.py`).
The batch is 12 rows whose input and target differ in text on 10, in language on 11 and
in length on 10. Inputs are 23 and 22 wide, targets 19 and 16.

| check | result |
|---|---|
| Batch tensors against a hand-written layout, by dataset and by tokenizer, pair and repetition | 40 of 40 equal in dtype, shape and values |
| `pipeline.forward(batch)` against `Model.forward` on tensors picked by hand | 40 of 40 logits bitwise equal: 5 pathways, 2 seeds, train and eval, pair and repetition |
| `compute_loss` against cross-entropy written out in float64 | 44 of 44 terms. Worst relative deviation 1.5e-07; the same sum in float32 against float64 gives 9.4e-08 |
| Metrics after 60 steps against counts by hand | 20 of 20 |
| Read set: roll one tensor along the batch. The loss moves exactly when the pathway reads it | 440 of 440 cells: 10 tensors, 5 pathways, train and eval, pair and repetition, the summed loss and each term. 0 cells move with nothing rolled |
| An input where a target belongs, and 16 other ill-formed constructions | 17 of 17 raise at construction or at the boundary |

Generation reads the input and `orth_start`, and nothing else
(`pr2/confirm2/r/accept/generate_reads.py`). Across 5 pathways, greedy and sampled:

- Changing the target's strings or its languages changes 0 leaves of the output in 10 of
  10 cells. Changing one input string changes the output in 10 of 10.
- The first token written is the language `orth_start` names on 8 of 8 rows, on each of
  the three pathways that write orthography. The two others do not read it.
- 37 of 37 refusals behave as designed, and all 9 missing-modality cases name the pathway
  and the modality. With the seed taken from the input, as at `7386f8e`, 4 of 8 rows open
  in the named language, and the probe flags it.

A model learns a target that is not its input. With `d_model` 64, `d_embedding` 8 (the
memory rows the decoder reads), batch 32 and 5000 steps, scored by greedy generation on
seeds 0, 1 and 2. Cells are the median with [min, max]. Seed 0 is
`pr2/confirm2/r/e2e/run_e2e.sh`. Seeds 1 and 2 are `pr2/docs2/e2e/run_seeds12.sh`, the
same scripts on the same library. `pr2/docs2/e2e/summary.out` holds all three.

| task | result |
|---|---|
| Next word: a 2 or 3 word phrase in, one word out, fixed by the phrase's last word. 300 held-out phrases, chance 0.017 | `o2o` 1.000 [1.000, 1.000]; `o2p` 0.997 [0.993, 0.997]; `p2o` 1.000 [0.997, 1.000]; `p2p` 0.997 [0.997, 1.000] |
| The same with targets permuted across rows | `o2o` 0.020 [0.010, 0.033]; `o2p` 0.000 [0.000, 0.000]; `p2o` 0.020 [0.013, 0.027]; `p2p` 0.000 [0.000, 0.000] |
| The output is the input's last word | 0.000 on held-out rows in every run with the real targets |
| Translation pairs both ways, `o2o`. 464 training rows scored | The right start gives 1.000. The start taken from the input gives 0.000 by ids, since the output opens with the wrong language token. Its text is still the target on 0.974 [0.918, 0.978] of rows |
| The same plus repetition rows: every input has two targets, its translation and itself, told apart only by `orth_start`. 928 training rows scored | The right start gives 1.000. The start taken from the input gives 0.500 [0.500, 0.500]. Both starts give their own target on 1.000 of inputs |
| Mixed-language target, `o2o`, 300 held-out rows, 225 switch tokens | 1.000 [1.000, 1.000] exact; every switch in the right place with the right id |
| The same on `p2o` | 1.000 [0.997, 1.000] exact; every switch right |

These are ceiling scores on small closed vocabularies. They show that an input and a
different target train and generate end to end, not how the model holds up on new words.

The scoring can see a wrong wiring. In a copy whose dataset encodes the input as the
target, the output is the whole input on 1.000 of held-out rows (seed 0). An id score that
takes its gold from `dataset[i].target` stays at 1.000 there, so the orthographic scores
are also taken against the file's own target strings, and agree.

## What is checked, and when

72 constructed dataset cases behave as the design says and none against it
(`pr2/confirm2/r/accept/dataset_boundaries.py`). The design is silent on 19 more, and
their outcomes are recorded. The ones a caller can meet:

- **A misspelt `target_raw` raises.** `target`, `Target_Raw` and a trailing space in the
  header each raise `KeyError` listing the columns found and naming the fix. A pkl without
  the key raises the same.
- **A misspelt `target_language` is not noticed.** Beside a single-code `language`, the
  column is ignored and the target takes the input's code. A target in the other language
  is then encoded in the wrong one, or raises when read if its words are missing. Beside
  a per-word `language` the load raises, since nothing can be copied.
- **A misspelt `language` is not noticed either.** Every word is English, as in 0015.
- **A blank `word_raw` or `target_raw` cell raises at load,** naming the row and the
  column, as a blank language cell does. A row one field short under a header ending in
  `target_raw` holds a blank target. An empty target would teach `[EOS]` straight after
  `[BOS]` (`pr2/docs2/empty_string.py`). `7386f8e` loaded a blank `word_raw` as an empty
  word and read it, in a csv with or without a language column and in a pkl
  (`pr2/confirm2/r/s6_regress/legit_vs_base.py`, `more_legit.py`).
- **Pronunciations are not checked at load.** Loading 200 rows calls the tokenizer 0
  times. Reading one row three times encodes each side three times.
- **A failed read names every row and side it could not pronounce.** It numbers a row by
  its position in `dataset.examples`, and a load error numbers the rows of the file.
  After `shuffle(seed=3)` a nonword at file row 5 is named as row 1
  (`pr2/confirm2/r/s6_regress/row_numbers.py`). The message quotes the string. Finding
  the rows encodes each one alone: a 7300-row slice holding one nonword takes 1.74 to
  1.94 s to fail, against 0.35 s to read clean (`more_legit.py`, three runs, load average
  about 10).
- **A shuffle moves whole rows.** 0 misaligned examples over 7 shuffles. Shuffling the
  inputs alone, as two parallel lists allow, misaligns 199 of 200, and the probe counts
  them.
- **A slice of a dataset that selects no rows raises `IndexError`,** naming the slice
  and the dataset's size. `single_step` and `evaluate` refuse an empty slice by name
  first.
- **The comma check covers a csv on disk.** The same typo on a `gs://` path loads the
  codes as the words (`pr2/confirm2/r/accept/extras.py`). The check reads the index of
  the frame pandas returns. A frame from a storage client may carry any index, and five
  such frames load as at `7386f8e` (`legit_vs_base.py`).
- **`orth_start` is checked for shape, dtype, device and id range only.** `[BOS, EN]`,
  `[EOS, PAD]`, two letters and `[EN, EN]` each pass and seed the decoder
  (`pr2/confirm2/r/s5_finding/declined/design_gaps.py`). The tokenizer's `orth_start`
  takes one code per row. Two codes in one entry raise the count message every entry
  point gives, `2 language codes for 0 words in ''` (`generate_reads.py`).
- **`BridgeInput` cannot tell its two modalities apart.** Both are integer `Tokens`, so
  orthography passed as `phon` builds and generates
  (`pr2/confirm2/r/s5_finding/declined/design_more.py`). Each is passed by name, so a lone
  modality cannot land in the other's field unnamed. `BridgeTarget` can: the label ranks
  differ.
- **A wrong side handed to the pipeline raises `AttributeError`.** `forward` given an
  input or a target, and `compute_loss` or `compute_metrics` given an input, each raise,
  4 of 4. The message names a missing attribute, not the mistake (`design_more.py`).
- **Logits of another width than the labels are torch's to refuse.** `compute_loss` given
  logits two positions narrower raises `RuntimeError`, with the message
  `Expected target size [8, 6], got [8, 8]`. `7386f8e` raised a `ValueError` naming both
  shapes (`pr2/confirm2/r/s6_regress/misuse_messages.py`). It takes logits and a target
  from different batches to reach it.
- **A call with no rows fails without naming the cause.** `encode_input([])` and
  `orth_start([])` raise `max() iterable argument is empty` (`extras.py`).

## Consequences

**Removed.** `EncodingComponent`, `BridgeEncoding`, the three `encode` methods,
`modality_filter`, the methods that built placeholder tensors, `padded_targets`,
`rows_and_languages`, the encoding memo, `_encode_single_word`, the `words` and
`languages` lists of a dataset, `_check_orth_target_width`, `orth_dec_prefix` and the
`word` key of an event. Each of 16 patterns is found in `bridge/` at `7386f8e` and not in
the working tree (`pr2/docs2/removed_names.py`). A batch has no `device` of its own,
which the batch type of `7386f8e` had (`pr2/confirm2/r/s6_regress/types_probe.py`). Its
`input` and its `target` each have one (`pr2/docs2/batch_device.py`).

**Every data file needs a `target_raw` column.** One file in the repo has a `word_raw`
column, `tests/domain/model/data/data.csv`. Its 7300 rows carry `target_raw` equal to
`word_raw`, and all 7300 load.

**The shift 0004 chose has one home.** The tokenizer builds the labels beside the decoder
ids. The loss and the metrics read `target.orth.labels` and slice nothing. The width
check `compute_loss` made is the `Target` type's: labels of another width cannot be
built. This amends where 0004 puts the alignment, not the alignment.

**The caller names the output language.** 0008 took the seed off the encoding, so the
output opened in the input's language. That is right for repetition and wrong for half of
a task whose target opens in the other language half the time: 0.500 above. The seed is
still `[LANG, BOS]`, the two positions training feeds the decoder first.

**The third failure of issue #233 is closed.** An input that lacks a modality its pathway
reads raises `pathway 'o2p' reads orth, which this input does not carry`, on 3 of 3
pathways that read orthography and, for orthography alone, on 3 of 3 that read
phonology. A zero-row batch raises by name on 5 of 5. Half precision is still open, and
narrower than the issue says: float16 and bfloat16 generate on `o2p` and `p2p`, and are
rejected with `orth_probs probabilities must sum to 1` on the three pathways that write
orthography (`pr2/docs2/issue_233.py`).

**Repetition tests cannot see the two sides wired the wrong way round**, because both
sides hold equal tensors. With the dataset's sides exchanged, the repetition section of
the training differential differs on 0 leaves on 5 of 5 pathways, and the pair section on
3040 to 10215 (`diff_training.py`). Two oracles in
`tests/application/training/test_input_target_wiring.py` catch it: the read set, and the
hand-wired reference. The reference has two tests. One compares the forward pass and
`evaluate`. The other, the step test, compares one optimizer step. Planted in copies of
the tree, each followed by the full suite:

| planted defect | suite | read set | hand-wired reference | script |
|---|---|---|---|---|
| None | 872 passed | | | `pr2/confirm2/r/accept/plant_suite.py` |
| The pipeline feeds the encoders the target and the decoders the input | 61 failed | fails on 5 of 5 pathways, on both batches | fails on 5 of 5, on both | same |
| The dataset encodes the target strings as the input and the input strings as the target | 41 failed | passes: the batch it builds is consistent | fails on 5 of 5, on both batches | same |
| The decoder is fed the input's ids and mask, only where the input is one token wider than the target | 15 failed | fails on 5 of 5, on the batch with repetition's widths | fails on 5 of 5, on that batch | same |
| The decoder's pad mask alone comes from the input, under the same width condition | 15 failed | fails on 5 of 5, on that batch | fails on 5 of 5, on that batch | `pr2/confirm2/r/s5_finding/plants.py`, plant `M211` |
| An optimizer step reports the true loss and trains on the batch with its sides exchanged | 10 failed | passes | the forward test passes; the step test fails 10 of 10 cells | `pr2/confirm2/r/s5_finding/step_plants.py`, plant `V13` |

The two width rows need a batch of their own. The width condition holds for repetition
and for no row of a batch whose sides differ in length, so the wiring tests run on two
batches. In one the sides differ in content, language and length. In the other each
row's target is the next row's input, so the sides differ in content and have
repetition's widths. In a copy of the tree without the second batch, four plants under
that width condition pass the whole suite (`pr2/docs2/gaps/table/plants.py --before`,
`plants_before.out`). On repetition the first width row moves logits and nothing else:
144 of 144 logits leaves and 24 of 212 leaves of `evaluate`, and 0 of 264 losses, 0 of
864 metrics and 0 of 20280 parameter leaves
(`pr2/docs2/gaps/accept/decfeed_by_group.py`).

The reference is wired from the rows as the test file writes them, not from the
dataset's. Its forward test covers the logits, the loss and the metrics `evaluate`
reports. Its step test covers the loss `single_step` reports and every parameter the step
leaves, compared bitwise.

The parameters show which loss the step ran `backward` on. In a copy of the tree whose
step test compares the reported loss alone, the last row of the table passes the whole
suite. So do three more of its kind: a second forward pass, twice the loss, and another
row's labels (`step_plants.py --before`, `step_plants_before.out`). In the working tree
all four fail 10 of 10 cells (`step_plants.out`). So do four more step defects: two
optimizer steps, a forward pass in eval mode, halved gradients, and encoder rows rolled
(`pr2/confirm2/r/s5_finding/my_step_variants.py`).

The step test passed 23 of 23 runs at 1, 2, 4, 12 and default thread counts, with CUDA
visible and hidden (`pr2/confirm2/r/s6_regress/step_test_stability.sh`). Comparing the
parameters costs 0.20 s: the test's ten cells take 1.10 to 1.12 s against 0.90 to 0.93 s,
over six alternating rounds. The whole suite moves by +0.16 s in those rounds, inside the
0.21 s that one copy varies by (`pr2/docs2/suite_time/suite_time.py`, `suite_time.out`).

Three more counts of planted defects, each one followed by the full suite:

| plants | result | script |
|---|---|---|
| 33, one for each gap a review found in the tests | 33 of 33 fail the test named as their guard. Of the 24 that fit the tree without those tests, 23 pass its suite | `pr2/confirm2/r/s5_finding/plants.py`, `plants.out`; `pr2/docs2/gaps/table/plants_before.out` |
| The test review's 140 | 137 still apply. 135 fail a test, 14 of which passed the suite without those tests. The 2 that pass are comment-only controls. The 3 whose lines changed are planted again among the 33 | `pr2/confirm2/r/s5_finding/rerun_mutants.py` |
| 18 variants of the fixed defects | 17 fail their guard. A `Target` that accepts uint8 labels beside int64 passes the whole suite | `pr2/confirm2/r/s5_finding/own_plants.py` |

No test pins that `Target` refuses uint8 labels. Such labels cannot train in silence:
`cross_entropy` refuses every integer dtype but int64 by itself (`more_legit.py`).

**Reading a slice encodes each side.** `dataset[slice]` takes 2.0 ms median at batch 128
against 1.3 ms at `7386f8e`, and 0.34 against 0.21 ms at batch 16. Over three interleaved
rounds, the median of a whole `o2p` step at batch 128 is 78.3 to 78.8 ms against 77.6 to
78.1, and of an `op2op` step 157.9 to 159.9 against 157.5 to 158.3. A step's own 10th to
90th percentile band is 1.4 to 3.4 ms wide on `o2p` and 3.0 to 12.4 ms on `op2op`, so the
0.7 ms a read adds is inside it (`pr2/docs2/timing/time_step.py`, `time_step.out`; 40
timed steps after 5 discarded, load average 1.5 to 2.0).

**The documents are held by tests.** `tests/test_documentation.py` runs every python block
of the README as written and compares each result the README shows, checks the map's
tensor listing against a live batch, and checks both type tables against the classes. 13
of 13 planted document defects fail their guard (`pr2/docs2/doc_test_plants.py`). Beyond
the suite, 93 sentences of the README and the map were each checked against the running
code, with 3 of 3 planted defects failing (`pr2/docs2/doc_claims.py`). Every example in
the README also runs as written with CUDA visible and with it hidden
(`pr2/docs2/readme_blocks.py`).

**The suite** is 873 passed and 6 skipped, against 711 and 6 at `7386f8e`. Over six runs
it takes 10.11 to 10.46 s by pytest's own count, against 7.04 to 7.48 s for `7386f8e`
with the two run alternately (`pr2/confirm2/r/s2_suite/suite_runs.sh`). A copy of the
tree outside git gives one fewer passed and one more skipped, because the test that asks
GitHub which issues are open skips there. That is the 872 of the planted runs above.

**A training event still names its rows by string.** It carries two strings per row, and
neither identifies a row.

Not measured:

- CUDA, for any number in this record.
- A trained model on new words or on real text.

Scripts and outputs are under `pr2/` in the evidence folder that
`docs/specs/input-target-pairs.md` names. `pr2/confirm2/r/` holds the confirming run of
every probe on the working tree. `pr2/docs2/` holds further runs on the same tree, most
of them the same scripts.
