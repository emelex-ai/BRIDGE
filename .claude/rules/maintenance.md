# Keeping the map and the constants true

`docs/architecture.md` and `.claude/rules/measurement.md` describe the current state. A stale
description is worse than none, because it gets trusted. Two mechanisms keep them true, and
they cover different things.

## What a test catches

`tests/test_documentation.py` asserts the machine-checkable claims against the live system:
the pathway table against `model.PATHWAYS`, the phoneme table dimensions and id-space bounds,
the recorded fingerprint against `phonreps.csv`, the documented target width, every path named
in the layout, the decision index against the files on disk, and whether the issues listed
under Known defects are still open.

Drift in any of those is a **failing test naming the file to edit**, not a document nobody
notices has gone wrong. Each planted defect was verified to fail the suite before the tests
were trusted.

## What a test cannot catch

Prose meaning. A module whose responsibility changed while its path stayed the same, a data
flow description that no longer matches the code, a decision that has quietly stopped holding.
For those the trigger is judgement, so the triggers are enumerated rather than left to a
general instruction to keep things updated.

## Update `docs/architecture.md` when

- a pathway is added, removed, or changes what it reads or writes
- a module is added, removed, renamed, or takes on a different responsibility
- a shape or dtype anywhere in the documented data flow changes
- an issue in Known defects is closed, or a defect of comparable severity is found
- a decision record is added, since the index lives here

## Update `.claude/rules/measurement.md` when

- `bridge/core/phonreps.csv` changes, which moves the fingerprint, the dimensions and both id
  ranges at once
- a noise floor is measured on a device or configuration not already recorded, most obviously
  the first CUDA sweep
- something listed under "What has not been measured" gets measured, whether or not the result
  is what was expected
- the oracle fixture is regenerated, which should approach never
- the commands for running the suite or the checks change

## Write a decision record when

A choice constrains future work and its reasoning is not recoverable from the code. Evidence
that overturns an earlier decision gets a **new** record that supersedes it; the old record is
marked, never edited, so the reasoning that applied at the time survives.

Records carry the numbers and name the script that produced them, per
`~/.claude/rules/measurement.md`.

## Definition of done

A change is not complete when the tests pass. It is complete when the tests pass and every
document describing what changed says something still true. If a measurement produced a number
these files record, the update lands in the same change as the measurement, not later.

An out-of-date fact that cannot be checked mechanically carries the date it was established and
the artifact that established it, so staleness is visible rather than silent.
