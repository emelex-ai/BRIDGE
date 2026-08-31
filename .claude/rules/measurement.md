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

## What has not been measured

**Numerical** equivalence work is still CPU only, as of 2026-08-30. CUDA differs in reduction
order and kernel selection, so the noise floor above does not transfer to the RTX 5080 until
re-measured there, and any claim of GPU equivalence is currently unsupported.

CUDA *behaviour* has been measured, on an RTX 5080 with torch 2.12.0+cu130: device
resolution and comparison, and that all five pathways generate on a GPU. That is placement
and control flow, not numerics, and the two should not be conflated.

`device_manager` no longer defaults to CPU unconditionally. It reads `BRIDGE_DEVICE`, so a
measurement script inherits whatever the shell exports. Record the device the run actually
used rather than assuming CPU. `tests/conftest.py` pops the variable so the suite is
unaffected, but probes are not covered by that.

## Two integer id spaces

Both are `torch.long` and nothing but naming separates them. Confusing them produces silently
wrong output rather than an error.

| space | range | what it indexes | where it appears |
|---|---|---|---|
| row | 0 to 90 | phoneme table rows (91 phonemes plus specials) | `EncodingComponent.enc_input_ids` / `dec_input_ids`, `Model.embed_phon_tokens` |
| feature | 0 to 35 | phonetic feature columns (31 base plus 5 specials) | `GenerationOutput.phon_tokens`, `PhonemeTokenizer.decode`, `VocabSpec.phon_*_id`, loss targets |

Current table fingerprint: `4c4a9c388f46bc35`. A checkpoint recording a different one was
trained against a different `phonreps.csv` and its phoneme ids may not mean the same things.

## Running things

```
uv run pytest -q
uv run mypy bridge
uv run ruff check . && uv run ruff format --check .
```

The suite runs in single-digit seconds because the pronunciation lexicon parse is memoised
process-wide. A change that pushes it past about 10 seconds has probably reintroduced
per-instance parsing, which cost 45 seconds before the memo (measured 2026-08-30).

## Probes

Investigation scripts belong in the session scratchpad and are expected to be thrown away.
Promote one into the repo only when it will be re-run against a future change, and when you
do, it belongs beside the harness it resembles rather than in a new directory:
`tests/fixtures/capture_phon_baseline.py` is the existing example.
