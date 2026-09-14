# BRIDGE

A dual-modality reading model: every word has an orthographic side (characters) and a
phonological side (phonemes), and the model learns to move between them.

`docs/architecture.md` is the map. Read it before answering structural questions, use its
vocabulary rather than inventing parallel terms, and update it when the structure changes.

@docs/architecture.md

Architecture decisions and the evidence behind them live in `docs/decisions/`, one immutable
numbered record each. The map carries an index of them. Read the relevant record before
revisiting a decision it covers, and add a new record rather than editing an old one.

Measurement constants for this repo, the baseline, the oracle, the noise floor and the two
phoneme id spaces, are in `.claude/rules/measurement.md` and load automatically.
