# Measurement constants for this repo

The global rules in `~/.claude/rules/measurement.md` say what counts as evidence. This file
says what the terms mean here, so the baseline, the oracle and the noise floor do not have to
be rediscovered every session.

## The oracle

`tests/fixtures/phon_baseline.pt` is a golden master: a behavioural fingerprint of the
phonological path recorded before the issue #221 refactor. `tests/domain/test_phon_baseline_equivalence.py`
asserts every recorded value against a freshly built model.

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

Every other phonological value in the fixture, which is what it was recorded for, is still
asserted exactly.

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

Established by perturbing something semantically irrelevant, not assumed. Running the same
training loop at `HEAD` against itself with `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` gives:

| comparison | max relative loss deviation | relative L2 parameter distance |
|---|---|---|
| HEAD vs HEAD, single-threaded BLAS | 2.229e-07 | 9.559e-06 |
| same tree, rerun | 0.0 | 0.0 |

So a change whose loss trajectory stays within roughly **2.2e-07** relative is inside the
floor. Measured 2026-08-30 on CPU, 12 Adam steps, `d_model=32`, batch 8, over the `o2p`,
`p2p`, `p2o` and `op2op` pathways. Re-measure rather than reuse this figure if the shapes,
the optimizer or the device change. Reruns of an unchanged tree are bit-exact, so any nonzero difference between two runs
of the same code indicates a real nondeterminism, not noise.

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
explicit `.to("cuda")`. See `docs/decisions/0012`.

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

CUDA *behaviour* has been measured, on an RTX 5080 with torch 2.12.0+cu130: device
resolution and comparison, that a constructed model is entirely on one device, and that all
five pathways generate on a GPU with no explicit `.to()`. That is placement and control flow,
not numerics, and the two should not be conflated.

Since #229, initialisation no longer depends on the device: weights are drawn on the default
device and then moved, so a seed gives bitwise identical parameters on CPU and CUDA, measured
169/169 at seed 5. A CPU-recorded fingerprint is therefore a valid starting point for a CUDA
run, which it was not before.

`device_manager` no longer defaults to CPU unconditionally. It reads `BRIDGE_DEVICE`, so a
measurement script inherits whatever the shell exports. Record the device the run actually
used rather than assuming CPU. `tests/conftest.py` pops the variable so the suite is
unaffected, but probes are not covered by that.

## Two integer id spaces

Both are `torch.long` and nothing but naming separates them. Confusing them produces silently
wrong output rather than an error.

| space | range | what it indexes | where it appears |
|---|---|---|---|
| row | 0 to 90 | phoneme table rows (91 phonemes plus specials) | the phonological `ids` of `BridgeInput.phon` and `BridgeTarget.phon`, `Model.embed_phon_tokens` |
| feature | 0 to 35 | phonetic feature columns (31 base plus 5 specials) | `GenerationOutput.phon_tokens`, `PhonemeTokenizer.decode`, `VocabSpec.phon_*_id`, the phonological `labels` of `BridgeTarget.phon` |

Current table fingerprint: `4c4a9c388f46bc35`. A checkpoint recording a different one was
trained against a different `phonreps.csv` and its phoneme ids may not mean the same things.

## Running things

```
uv run pytest -q
uv run mypy bridge
uv run ruff check . && uv run ruff format --check .
```

The suite runs in about ten seconds because the pronunciation lexicon parse is memoised
process-wide. A change that takes it well past that has probably reintroduced
per-instance parsing, which cost 45 seconds before the memo (measured 2026-08-30).
Currently 873 passed and 6 skipped, in 10.11 to 10.46 s by pytest's own count, median
10.22, and 12.91 to 13.37 s of wall clock, which includes interpreter start. Measured
2026-11-23 on the input and target change: six runs of
`uv run --no-sync pytest -q -p no:cacheprovider` with `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`, load average 1.65 to 2.76. The 711 tests of `7386f8e` took 7.04 to
7.48 s by pytest's count, median 7.24, in the same sessions, the two suites run
alternately. That is 162 more tests and about 3 s more than `7386f8e`. Record 0016 names
the script. The suite took 12.9 s before the pipeline stopped running `gc.collect()` every
ten steps.

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
