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

It covers `README.md` too, as of 2026-09-16: its import list against `bridge.__all__`, its
worked example against `inspect.signature(TrainingPipeline.__init__)`, its pathway list, its
id-space bounds, its link targets, and its dependency sentence against `pyproject.toml`. The
README was outside the net until it had drifted four ways at once and its only worked example
raised `TypeError`. Each of those four assertions was verified to fail against the defect it
describes before being trusted.

Since 2026-11-22 it also runs the README's multilingual example as written, and compares the
result the example shows with what the call returns. While the per-word language change
was in progress the suite stayed green as that example raised `TypeError`. The check was
verified to fail on three planted defects: a keyword `encode` does not take, a wrong result
shown, and a word tagged with a language whose lexicon lacks it.

Since 2026-11-23 it also checks three things the input and target change wrote down:

- Every python block of the README runs as written, each alone, and every result the README shows equals what the call returns. The usage example trains on the csv block under "Dataset files", so that block is checked as a file that loads.
- The map's tensor listing names exactly the tensors of a live `BridgeBatch` (an input and a target, as the dataset returns them), with the rank and dtype of each.
- The type table in the map and the one in the README name every field of the five types.

Each was verified to fail on planted defects in a copy of the tree, 13 in all: a shown result changed, a call to a removed method, the data file without its target columns, an example using an undefined name, a tensor renamed or dropped from the listing, a mask listed as `int64`, a label width off by one, a field missing from a table row.

Since 2026-11-24 it also checks three things the position change wrote down:

- The call the map writes out for `Model.generate` is the method's signature: every parameter, its default, and which are keyword-only.
- The table of sizes in the map and the one in the README give every size of `ModelConfig` with its default.
- The parameter tensors and scalars `.claude/rules/measurement.md` records for a default model are a default model's.

Each was verified to fail on planted defects in a copy of the tree, 9 in all: the call shown without `max_new_tokens`, with a default for it, and with it positional; a `d_embedding` default of 1 in either table, a size missing from the README's, and a size in the map's that `ModelConfig` does not have; a tensor count of 169, and a scalar count of 629555. The check that runs the README's examples was verified on three more: a `generate` call without its cap in the generation example and in the usage example, and a changed result under the cap example.

`tests/test_diagram.py` does the same for `BRIDGE-Models.drawio`, the public architecture
figure, as of 2026-11-07. It checks one page per pathway; block opacity against which modules
each pathway actually fires, measured by forward hooks; the label of the block fed to the
mixer; the embedding cubes' axis labels against measured tensor shapes; every arrow attached
at both ends and animated exactly when active; and every "+" wired, with none feeding a
decoder, since the causal mask goes to self-attention rather than into a sum. As delivered,
the figure failed five of those ways at once, and each check was verified to fail against
that original.

Since 2026-11-08 it also checks three more things:

- Every block shared between pages sits at the same frame-relative position on each, within half a unit, so switching pages does not jump.
- Every decoder on every page is fed by an "Autoregressive Mask" that fades with it, against a live measurement that every decoder call is causal.
- Every page adds the global embedding back after the mixer, against a live bitwise measurement that the code does.

Each was verified to fail on a planted defect: a block nudged 2 units, and the global embedding's arrow cut from its "+".

The output box's feature table, the 0/1 values for "cat", is checked against the decoder's training targets, recomputed from the live tokenizer. An edit to `phonreps.csv` that changes those values fails the test. This was verified by flipping digits in a copy.

Since 2026-11-23 it also checks three things the input and target change drew:

- Every encoder is drawn reading the input box of its modality. Every decoder is drawn fed from the target box of its modality, through one "Tokenization + Embedding" block. Both are held against a live measurement of which side each one is fed, taken by hooks on the encoders and decoders.
- The cards of words are the batch that measurement runs on. The input boxes show the input words and their pronunciations from the live tokenizer, the target box shows the target words, no input word is its own target, and one header names each side.
- An input lane (the box of words and the tokenizer of one modality) is faded exactly when the page's pathway does not read that modality. A decoder's embedding block is faded exactly when its decoder does not run.

Each was verified to fail on planted defects in copies of the figure, 15 in all: a decoder fed from an input box, from the other modality's target box, with no embedding block or with its arrow deleted; an encoder fed from a target box; a wrong pronunciation card, an input card showing its target, a target card showing its input and the two headers swapped; an unread input box or tokenizer drawn opaque, and a decoder's embedding block drawn in the wrong state. The figure as it stood before that change fails all three checks. The first also fails on the current figure when `TrainingPipeline.forward` is patched to feed the decoders the input, so the measured half can see the code change.

Since 2026-11-24 it also checks where the figure says position enters. The two groups that hold the encoders and the decoders name rotary position, and no other label names position at all. That is held against a live measurement: each stage is handed its sequence reversed, and only the self-attention of the encoders and decoders may move.

It was verified to fail on 12 planted defects. Seven were in copies of the figure: "+ Position" back on an embedding label or on a decoder's feed, "(rotary position)" dropped from a group, added to the mixer or moved behind cross-attention, the two group labels exchanged, and an arrow note naming position. Five were in the code: the rotation switched off, a position term added in an embedder, and a slot signal fed to the mixer, to a cross-attention between modalities, and to a decoder's cross-attention. The figure as it stood before that change fails the check 25 ways.

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
- a dataset column is added or removed, or what a missing column means changes; the README's
  "Dataset files" section says the same thing and moves with it
- what `Model.generate` takes from its caller changes, such as `orth_start` or
  `max_new_tokens`; the README's "Generating" section says the same thing and moves with it
- where position enters the model changes, or what bounds a sequence's length
- a default in `ModelConfig` changes, or a rule relating two of its sizes
- what a checkpoint load accepts or refuses changes
- an issue in Known defects is closed, or a defect of comparable severity is found
- a decision record is added, since the index lives here

## Update `BRIDGE-Models.drawio` when

- a module is added to or removed from any pathway's data path; the test catches a block drawn
  in the wrong state, but not a module the figure never drew
- an operation inside a drawn block changes, such as a residual, a norm or a mask
- a label in `tests/test_diagram.py`'s block tables is renamed, in the same change
- where position enters the model changes, since two group labels say where it does
- `bridge/core/phonreps.csv` changes. Run `uv run python tests/fixtures/sync_diagram_table.py`, which rewrites the output boxes' feature table from the live targets and leaves everything else untouched; on an up-to-date file it changes nothing.

Render every page after editing and look at it; the test does not see layout.

## Update `.claude/rules/measurement.md` when

- `bridge/core/phonreps.csv` changes, which moves the fingerprint, the dimensions and both id
  ranges at once
- a noise floor is measured on a device or configuration not already recorded, most obviously
  the first CUDA sweep
- something listed under "What has not been measured" gets measured, whether or not the result
  is what was expected
- the oracle fixture is regenerated, which should approach never
- the commands for running the suite or the checks change
- the suite's runtime moves enough to change the figure it gives for the suite
- the model gains or loses a parameter tensor, which moves the count it records

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
