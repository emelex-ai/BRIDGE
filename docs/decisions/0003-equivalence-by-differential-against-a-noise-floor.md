# 0003. Behavioural equivalence is proven by differential against a measured noise floor

Status: Accepted
Date: 2026-08-30

## Context

Refactors of the model must not change what it computes. The test suite cannot establish that:
it asserts the properties someone thought to write down, and a refactor can pass every test
while changing numerics.

Bitwise identity is the wrong bar. Floating point addition is not associative, so reduction
order changes results, and reduction order changes with BLAS blocking, thread count, batch
shape and kernel choice. Demanding bitwise equality produces false alarms; accepting an
arbitrary tolerance proves nothing.

## Decision

Equivalence is established by running the same probe under a pristine `git worktree` at the
pre-change commit and under the working tree, then comparing, with three requirements.

**Split exact from approximate.** Integer ids, boolean masks, generated token sequences and
canonicalised feature sets must match bitwise. Only floats get a tolerance, normalised by
tensor scale rather than per element, since per-element relative error diverges near zero and
reports artifacts as failures.

**Establish the noise floor empirically.** Perturb something semantically irrelevant, such as
BLAS thread count, and measure the resulting deviation. That is the floor. A change is
equivalent when it does not exceed it. Without this number the measured deviation is
uninterpretable.

**Compare the stable observable.** Optimizers normalise by gradient magnitude, so a 1e-07
forward difference becomes a 1e-02 parameter difference after a few dozen steps while the
system remains equivalent. Judge on the loss trajectory and explain parameter divergence rather
than reporting it as failure.

Separately, `tests/fixtures/phon_baseline.pt` pins a golden master of the phonological path in
CI. It is immutable, and its provenance was verified by regenerating it from a pristine
worktree and diffing: byte-identical.

## Evidence

The method caught what the suite did not, and its own controls caught three instrument bugs:
a comparator using per-element relative error reported 5,006 false failures; a silently broken
logger produced an empty file that would have read as a clean result; and copied credentials
expiring mid-run left 11 of 24 samples missing, unevenly across arms.

The suite passed throughout. Every one of those would have produced a confident wrong
conclusion.

## Consequences

A refactor of the model is not done when the suite is green. It is done when a differential
against a pristine baseline shows exact-valued quantities unchanged and float deviation inside
a measured floor.

This is slower than running tests, and the cost is real. It is also the only method here that
has ever detected the difference between "the tests still pass" and "the model still computes
the same thing".

The current floor is CPU only. Extending any equivalence claim to CUDA requires re-running the
differential there.
