# Measurement constants for this repo

The global rules in `~/.claude/rules/measurement.md` say what counts as evidence. This file
says what the terms mean here, so the baseline, the oracle and the noise floor do not have to
be rediscovered every session.

## The oracle

`tests/fixtures/phon_baseline.pt` is a golden master: a behavioural fingerprint of the
phonological path recorded before the issue #221 refactor. `tests/domain/test_phon_baseline_equivalence.py`
asserts what the recording still says of a freshly built model. The table below says how
much that is.

**It is immutable.** Regenerating it to make a test pass destroys the only thing it is for. It
was verified genuine by rebuilding it from a pristine worktree at `5d7e8d9` and diffing:
byte-identical, zero differences. Regenerate only by running
`tests/fixtures/capture_phon_baseline.py` on a pre-refactor commit.

**Part of it is now superseded, and the file was still not touched.** Issue #228 deliberately
changed how the orthographic decoder is seeded at generation, so the recorded `orth_tokens`
and `orth_probs` describe behaviour that was wrong. The handling is the pattern to copy when
this happens again: measure the change, bound it, reduce the superseded assertions to what
still holds, and pin the new behaviour somewhere else against analytic oracles rather than
against a re-recording. Measured 2026-09-14, `global_encoding` unchanged on all five
pathways and every phonological value unchanged including `op2op`'s, which generates both
modalities from one encoding; only the three orthographic pathways moved, at identical
shapes. See `docs/decisions/0008`.

**A second value is superseded the same way.** The fixture records `placeholder_phon`, the
stand-in phonology an orthography-only encoding carried. An input with no phonology carries
`None`, so there is no tensor to compare. The test asserts what the stand-in was for: an
input carrying orthography alone generates what the fixture records for `o2p`. Measured
2026-11-23 against `7386f8e`, generation from such an input is bitwise equal to generation
from the stand-in on `o2p` and `o2o`, 729 of 729 values. See `docs/decisions/0016`.

**Every value that passes through the model's weights is superseded too, and the file
was still not touched.** Position is a rotation inside self-attention, and the model
holds no position table (`docs/decisions/0017`). The recording was made by a model with
two tables of 30 rows. Without them a seed's draws fall differently across the parameters
that remain, so no recorded embedding, logit, encoding or generated token can be produced
again. Measured 2026-11-24 against `a3cb285`: with that commit's tables zeroed and the
rotation switched off, the same weights give bitwise equal logits on all five pathways,
551424 elements, so position is all that moved.

**One vocabulary value is superseded as well, and the file was still not touched.** The
fixture records `orth_vocab_size` 109. The tokenizer's vocabulary is those 109 entries
followed by 16 characters of Spanish (issue #237), so its size is 125. The vocabulary node
reads the recorded 109 exactly, as the size before the characters `SPANISH_CHARACTERS`
names, and every other recorded value as it did. Measured 2026-11-27 against `3d28ee9`,
on CPU: none of the 109 entries moved, and every row that decoded to its own text there
encodes to the same ids, masks and labels, 263264 of 263264 rows and 46390938 elements,
over both lexicons, each tagged with its language and untagged, and 4818 mixed phrases.
Each recorded vocabulary value changed by one, up and down, fails the node 16 times of
20 on both trees. The four that pass change `orth_spc_id` and `phon_spc_id`, which
`VocabSpec` does not carry. The other 23 nodes are untouched and pass: the model they
rebuild is built at the recorded size, which every id of the fixture's words is under.
The ids of the 16 characters are pinned by hand in
`tests/domain/tokenizer/test_character_tokenizer.py`. Scripts:
`spanish-patching/letters/03_ids_compare.py` and `10_baseline_vocab_mutations.py`.

What the 24 test nodes still assert. The table was measured 2026-11-24 by mutating a copy
of the recording 255 ways and running the nodes of both trees on each, and its vocabulary
row again on 2026-11-27 as above:

| nodes | how they read the recording |
|---|---|
| 9 tokenizer | exactly, as at `a3cb285` |
| 1 vocabulary | exactly; `orth_vocab_size` as the size before the 16 Spanish characters |
| 1 gradient of the phonological embedding | at the tolerance it has always had; no weight enters that gradient |
| 1 gradient of the position table | exactly: the recording is the count of rows per slot, and no parameter is left for it to reach |
| 1 input without phonology | exactly on the recorded stand-in; its generation is compared with an input carrying both modalities |
| 3 generations that write phonology | form, the recorded `[BOS]` that opens each sequence, and each recorded length |
| 2 embeddings, 4 logits, 2 generations that write orthography alone | form only: which outputs are filled, shape, dtype, finiteness, id range |

That is 10 nodes reading the recording as they did and 14 changed. None is deleted and
none is emptied: each still fails on some change to the recording. What the model
computes is pinned against reference and analytic oracles in
`tests/domain/model/test_rotary_layers.py`, `tests/domain/model/test_any_length.py` and
`tests/domain/model/test_max_new_tokens.py`, not against a re-recording.

**The phoneme table has seven more rows than the recording saw, and no node moved.** The
table holds 98 rows, with seven Spanish phonemes at rows 91 to 97. All 24 nodes pass with
the test file and the fixture untouched. Three things make that so. The recording holds
phonology as feature sets, not row ids. Its nine words are English. Every row those
words use kept its id and its features. Measured 2026-11-27 on CPU against `3d28ee9`,
one thread, torch 2.12.0+cu130:

| what | reading | script |
|---|---|---|
| every lexicon word that encoded without `[UNK]` at `3d28ee9`, encoded on both trees | 127925 of 127925 words bitwise equal, 1066293 ids | `spanish-patching/phonemes/compare_encodings.py` |
| a model at seed 0, `d_model=32`: parameters, 24 steps on each of five pathways, a generation on each | 1013 of 1013 float tensors and 635 of 635 integer tensors bitwise equal | `spanish-patching/phonemes/compare_model_differential.py` |
| the 24 nodes with `K`, a row the recorded words use, given `voice` in a copy | 4 fail | `spanish-patching/phonemes/plant_defects.py` |
| the 24 nodes with the special tokens moved to rows 93 to 97 in a copy | 0 fail: the recording cannot see a row id | the same |

Both comparators were checked on a planted difference first: one id moved by one, and one
float moved by one unit in the last place.

The fixture names its tensors as the code did when it was recorded: `enc_input_ids`,
`dec_input_ids`, `targets`. The test maps each to the tensor that holds it. The encoder's
are the ids and mask of the input (`BridgeInput`, what the encoders read). The decoder's
are the ids and mask of the target (`BridgeTarget`, what the decoders are fed and scored
against). The recorded targets are the target's `labels`. The fixture's words are encoded
once as an input and once as a target, which is repetition, the task it recorded.

## The baseline for a differential

A pristine `git worktree` at the commit before the change. The working venv resolves the
worktree's code ahead of the editable install when invoked as
`cd <worktree> && PYTHONPATH=$PWD <repo>/.venv/bin/python <script>`, so no second environment
is needed.

## The noise floor

Established by perturbing something semantically irrelevant, not assumed. The same
training loop is run twice on one tree, at two BLAS thread counts. Each figure belongs to
one model, so the table names it.

| model | comparison | max relative loss deviation | relative L2 parameter distance |
|---|---|---|---|
| this model: an orthographic vocabulary of 125, 167 parameter tensors | one thread against 6 | 2.309e-07 | 5.049e-06 |
| this model | one thread against 4 | 1.260e-07 | 4.826e-06 |
| this model | rerun at one thread | 0.0 | 0.0 |
| `3d28ee9`: an orthographic vocabulary of 109, rotary position, 167 tensors | one thread against 6 | 1.850e-07 | 6.223e-06 |
| `3d28ee9` | one thread against 4 | 1.480e-07 | 7.279e-06 |
| `3d28ee9` | rerun at one thread | 0.0 | 0.0 |
| `a3cb285`: position tables, 169 tensors | one thread against 6 | 1.327e-07 | 5.856e-06 |
| `a3cb285` | one thread against 4 | 1.327e-07 | 6.769e-06 |
| `a3cb285` | rerun at one thread | 0.0 | 0.0 |
| the tree of 2026-08-30: position tables | another BLAS thread count | 2.229e-07 | 9.559e-06 |
| the tree of 2026-08-30 | rerun | 0.0 | 0.0 |

So a change to this model whose loss trajectory stays within roughly **2.3e-07** relative
is inside the floor. The first nine rows share one protocol: 12 steps of `single_step`,
`d_model=32`, `nhead=2`, the default `d_embedding` (2 for this model and `3d28ee9`, 1 at
`a3cb285`), seed 0, batch 8, over the `o2p`, `p2p`, `p2o` and `op2op` pathways, torch
2.12.0+cu130, on CPU, on a machine whose default is 6 threads. Record 0017 names the
script. The first three were measured 2026-11-27 with that script unmodified
(`spanish-patching/letters/13_noise_floor.sh` runs it). The six below them were measured
2026-11-24, and the three of `3d28ee9` came out digit for digit the same when run again
in the session of 2026-11-27. The last two rows were measured 2026-08-30 on CPU: 12 Adam
steps, `d_model=32`, batch 8, the same four pathways.

The instrument was checked on this model. `global_embedding` scaled by 1 + 1e-3 reads
4.704e-05 and 3.034e-04, with 0 of 12 losses bitwise equal on every pathway. Scaled by
1 + 1e-6 it reads 1.947e-07 and 6.986e-06, which is inside the floor: a change that small
cannot be told from a thread count.

Re-measure rather than reuse these figures if the shapes, the optimizer or the device
change. Reruns of an unchanged tree are bit-exact, 167 of 167 tensors, so any nonzero
difference between two runs of the same code indicates a real nondeterminism, not noise.

A default model has 167 parameter tensors and 627843 scalars, at `d_embedding` 2 and an
orthographic vocabulary of 125.

## Splitting exact from approximate

Compare integer ids, boolean masks, generated token sequences and canonicalised phoneme
feature sets **bitwise**. Only float tensors get a tolerance, normalised by tensor scale
rather than per element. A mixed pass/fail over the whole structure hides which kind failed.

## Measured on 2026-09-16, during the simplification pass

Four numbers worth not rediscovering, all CPU, torch 2.12.0+cu130, commit `7979d5e`:

| what | reading |
|---|---|
| `gc.collect()` with the lexicon loaded | 155 ms median (768k tracked objects) |
| the same after `gc.freeze()` | 0.0 ms |
| the old per-ten-step `gc.collect()` | 17% of every epoch, freeing nothing (flat RSS control) |
| undetached loss tensors in `TrainingEvent` | +581 MB steady state, +527 MB/epoch if kept |

`TrainingPipeline` used to call `model.to(device_manager.device)` in `__init__`, which
silently moved a model the caller had placed: 169/169 parameters back on CPU after an
explicit `.to("cuda")`. The model of that commit had 169 parameter tensors. See
`docs/decisions/0012`.

The phonological metric helpers were verified equivalent across the dedup by differential
against the committed implementation, paired inputs, 6 seeds x 8 metrics, max relative
deviation 0.000e+00. The differential was validated by planting two defects and confirming
both were caught.

## What has not been measured

**Numerical** equivalence work is still CPU only, as of 2026-08-30. CUDA differs in reduction
order and kernel selection, so the noise floor above does not transfer to the RTX 5080 until
re-measured there, and any claim of GPU equivalence is currently unsupported. Every number
in `docs/decisions/0015`, the per-word language record, is CPU only too, as of 2026-11-22.
So is every number in `docs/decisions/0016`, the input and target record, as of 2026-11-23.
So is every number in `docs/decisions/0017`, the position record, as of 2026-11-24. So
is every number of the Spanish letters change, issue #237, as of 2026-11-27, and that
includes the noise floor of this model.

CUDA *behaviour* has been measured, on an RTX 5080 with torch 2.12.0+cu130: device
resolution and comparison, that a constructed model is entirely on one device, and that all
five pathways generate on a GPU with no explicit `.to()`. That is placement and control flow,
not numerics, and the two should not be conflated.

The same was measured again on this model, with rotary position, on 2026-11-24, RTX 5080,
torch 2.12.0+cu130. A model built on the process device sits on `cuda:0`. All five
pathways generate with no explicit `.to()`. One optimizer step and one `evaluate` give a
finite loss on each. A batch holding a 102-token row steps, evaluates and generates.
Record 0017 names the script. No CUDA float was compared with a CPU float.

And again on this model, with 125 orthographic entries, on 2026-11-27, RTX 5080, torch
2.12.0+cu130. The 1344 accented Spanish lexicon words encode on `cuda:0` to the ids the
vocabulary list gives, 1344 of 1344 rows, none holding `[UNK]`, and their decomposed
spellings to the same ids in both modalities. A model built on the process device sits
on `cuda:0`. All five pathways generate from a batch of eight of those words with no
explicit `.to()`, and one optimizer step on that batch gives a finite loss on each.
Script: `spanish-patching/letters/14_cuda_behaviour.py`. No CUDA float was compared with
a CPU float.

Since #229, initialisation no longer depends on the device: weights are drawn on the default
device and then moved, so a seed gives bitwise identical parameters on CPU and CUDA, measured
167/167 at seed 5 on 2026-11-27, on this model. Two seeds on one device differ in 81 of
167, the rest being biases and norm weights that start at 0 or 1. A CPU-recorded
fingerprint is therefore a valid starting point for a CUDA run, which it was not before.

`device_manager` no longer defaults to CPU unconditionally. It reads `BRIDGE_DEVICE`, so a
measurement script inherits whatever the shell exports. Record the device the run actually
used rather than assuming CPU. `tests/conftest.py` pops the variable so the suite is
unaffected, but probes are not covered by that.

## Two integer id spaces

Both are `torch.long` and nothing but naming separates them. Confusing them produces silently
wrong output rather than an error.

| space | range | what it indexes | where it appears |
|---|---|---|---|
| row | 0 to 97 | phoneme table rows (93 phonemes and 5 specials) | the phonological `ids` of `BridgeInput.phon` and `BridgeTarget.phon`, `Model.embed_phon_tokens` |
| feature | 0 to 35 | phonetic feature columns (31 base plus 5 specials) | `GenerationOutput.phon_tokens`, `PhonemeTokenizer.decode`, `VocabSpec.phon_*_id`, the phonological `labels` of `BridgeTarget.phon` |

Current table fingerprint: `1a9b6eee9bf74384`. A checkpoint recording a different one was
trained against a different `phonreps.csv` and its phoneme ids may not mean the same things.

## Running things

```
uv run pytest -q
uv run mypy bridge
uv run ruff check . && uv run ruff format --check .
```

The suite runs in about eleven and a half seconds because the pronunciation lexicon parse
is memoised process-wide. A change that takes it well past that has probably reintroduced
per-instance parsing, which cost 45 seconds before the memo (measured 2026-08-30).
Currently 1046 passed and 6 skipped, in 11.39 to 11.55 s by pytest's own count and 14.28
to 14.43 s of wall clock, which includes interpreter start. Measured 2026-11-24 on the
position change: three runs of `uv run --no-sync pytest -q -p no:cacheprovider` with
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, load average 1.4 to 2.3. The 873 tests of
`a3cb285` took 10.07 to 10.22 s by pytest's count in the same session, the two suites run
alternately. That is 173 more tests and about 1.3 s more than `a3cb285`. Record 0017
names the script. The 6 skips are six cases of one test, each a pathway that does read
the modality the case leaves out. No test is skipped for want of a GPU on a machine that
has one. With the thread variables unset, 1042 of these tests were run and passed, so no
tolerance in them is tuned to one thread. The suite took 12.9 s before the pipeline
stopped running `gc.collect()` every ten steps.

`mypy` is clean over `bridge/`. CI enforces it rather than advising, as of 2026-09-16.

## Probes

Investigation scripts belong in the session scratchpad and are expected to be thrown away.
Promote one into the repo only when it will be re-run against a future change, and when you
do, it belongs beside the harness it resembles rather than in a new directory:
`tests/fixtures/capture_phon_baseline.py` and `tests/fixtures/bench_phon.py` are the two
that earned it. Named `capture_`/`bench_` rather than `test_`, so pytest does not collect
them. `tests/fixtures/sync_diagram_table.py` is a third, re-run whenever `phonreps.csv`
changes. It was verified on 2026-11-08 on files in its own format and on files re-saved by draw.io:

- It is a byte-identical no-op on a current file.
- It restores a corrupted table, with no duplicate cells left behind.
- It reshapes the table correctly under a simulated change to the targets.
