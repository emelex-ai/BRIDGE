# 0015. Language belongs to each word of a row

Status: Accepted, amended by [0016](0016-a-training-example-is-an-input-and-a-target.md)
and [0017](0017-positions-are-computed-for-any-length.md)

> The rules for languages this record chose still hold. Four details are amended. The
> tokenizer's calls are `encode_input` and `encode_target`, and a word with no
> pronunciation gives an input whose `phon` is `None`, or no target. `languages` is one
> entry per row and nothing else. A first row with no pronunciation loads and raises when
> it is read, like any other row. A phonology-only input has no orthographic placeholder,
> and the caller names the language orthographic generation starts in. See 0016.
>
> One more detail is amended. `VocabSpec.orth_language_ids` is required, so the
> `_without_switches` accuracy pair is always reported and is never absent. See 0017.
Date: 2026-11-22
Amends [0008](0008-generation-seeds-the-training-prefix.md) and one sentence of
[0004](0004-orthographic-teacher-forcing-alignment.md)

## Context

A row's language reached the tokenizers as `language_map`, a dict keyed by string. The two
tokenizers read it differently. `CharacterTokenizer` (characters to ids) looked up the
whole row. `PhonemeTokenizer` (words to phoneme ids) looked up each word of the row. The
words of a phrase are not keys of a map built from rows, so they fell back to English.

Ten defects followed from that one mismatch, reproduced at `9e7d9f2`, CPU, torch
2.12.0+cu130 (`wf_language/defects.py`). The three that matter most:

| defect | observed |
|---|---|
| A phrase tagged Spanish gets English phonemes | "no me" tagged ES encoded as `[86, 69, 33, 89, 68, 32, 87]`, the English rows. Spanish is `[86, 69, 22, 89, 68, 27, 87]` |
| A row's phonemes depend on its batch neighbours | "no me" tagged EN, beside a row "no" tagged ES, got a Spanish "no" |
| One string under two languages collapses in a slice | rows "no"/EN and "no"/ES both came out Spanish |

No test caught any of them, because no test compared a phoneme id against a language.

## Decision

**Every row carries one language code per word, and both tokenizers honour it by
position.** `bridge/domain/tokenizer/languages.py` holds the rules, and the tokenizers and
the dataset all read a row through it.

| point | rule |
|---|---|
| What a word is | A run of non-whitespace, `\S+`, one function for both tokenizers |
| Whitespace | Orthography keeps every character exactly as written |
| Codes | `EN` or `ES`, one per word. A single code covers every word |
| Code count | One, or the number of words. Anything else raises `ValueError` naming the row |
| Unsupported code | `ValueError`, one message from every entry point |
| No language | `--` for the whole row, with English phonemes. A tokenizer call with no language gives it. `--` beside a real code raises |
| Phonology | Each word is looked up in its own language's lexicon, by its position in the row. A word missing from it has no pronunciation; no other lexicon is tried |
| Orthography | The row opens with its first word's language token, then `[BOS]`. A language token appears again only where the language changes, directly before that word's first character |
| Decoding | `decode_with_languages` returns the text and each word's language, the one in force at its first character. `--` comes back as `None` |
| Data file | A `language` cell is `EN` or `EN EN ES`; a pkl cell may be a list or tuple. Stored as a tuple per row |

```
the word is gato        EN EN EN ES
[EN] [BOS] t h e _ w o r d _ i s _ [ES] g a t o [EOS]
```

The language tokens were already entries in the orthographic vocabulary, ids 6, 7 and 8 of
109, so the model does not change and a mid-row switch is a target the decoder learns to
emit.

`language_map`, `_resolve_language_map`, `_word_languages`, `strict_conflicts`, string
indexing (`ds["cat"]`) and `get_encoding` are removed. Each existed to find a language by
string, and one string can sit under two languages.

## What a row in one language keeps

Every row in one language that the code at `9e7d9f2` encoded correctly encodes bitwise
identically, and training, loss, metrics and generation on such rows are bitwise identical.
Measured on 2026-11-22 against a pristine worktree of `9e7d9f2`, CPU, float32, one thread,
torch 2.12.0+cu130. Integer ids and masks are compared with `torch.equal` plus dtype and
shape.

| check | result | script |
|---|---|---|
| Tokenizer and dataset: words, phrases, batches, odd whitespace, capitals, a custom dictionary, the 7300-row fixture | 28806 of 28806 identical | `pr1/confirm/accept/diff_tokenizer.py` |
| Five pathways, 200 rows, 8 steps, two model seeds: logits, losses, metrics, 169 parameters after each step, generation | 46975 of 46975 values bitwise | `pr1/confirm/accept/diff_training.py` |
| Control: the baseline against itself, and the working tree against itself | bitwise, 46975 of 46975 and 47095 of 47095 | same |
| Planted: one row retagged EN to ES | caught on 200 of 200 tokenizer cases, and on every pathway that reads the changed tensor | both |
| A csv of 320 rows through the dataset, 12 steps with metrics, `evaluate` and `generate`, five pathways | 9392 exact values with 0 differing; 5300 of 5300 floats bitwise | `pr1/confirm/e2e/diff_single.py` |
| Its control and its plant, one row retagged ES | tree against itself bitwise; the plant moves 25 exact values and 3536 of 5300 floats | same |
| A `p2o` checkpoint saved at `9e7d9f2`, loaded on this source | loss, both accuracies, generated ids and global encoding bitwise equal | `pr1/confirm/e2e/ckpt_roundtrip.py` |
| The golden master, `tests/fixtures/phon_baseline.pt` (the recorded fingerprint of the phonological path) | file byte-identical to `9e7d9f2`; its 24 tests pass | `pr1/confirm/04_golden_master_and_baseline_suite.out` |

The tokenizer differential also ran 1881 rows from the 10 classes expected to differ. Its
output lists them one per line, each with the defect that explains it, and reports no
difference it cannot explain. Three kinds of single-language row change on purpose:

- **A phonology-only encoding opens with its language.** The orthographic placeholder is
  `[LANG, BOS]` with the first word's language, where 0008 fixed it at `[--, BOS]`. `p2o`
  generation is seeded from it, so a row tagged EN or ES is seeded with that language.
  0008 called `--` the honest default because a phonology-only encoding did not say which
  language to spell in. It does when the caller passes one. The placeholder reads `[7, 0]`
  for EN and `[8, 0]` for ES, where `9e7d9f2` gave `[6, 0]` for both
  (`pr1/confirm/findings/rc/p05_placeholder.py`).
- **A word tagged with a language that lacks it does not encode.** "the" tagged ES has no
  Spanish entry and returned English phonemes. It returns `None`, like any word with no
  pronunciation.
- **Rows the baseline encoded wrongly** are fixed. The ten defects were checked as 17
  items: 11 are fixed, 5 are removed with the API that carried them, and 1 is kept on
  purpose, the English phonemes of a row with no language
  (`pr1/confirm/accept/defects.py`). One line of that output reads `README.md ... run as
  written -> TypeError`. It runs the example of `9e7d9f2`, typed into the probe, and does
  not read the file. The README's example runs: `tests/test_documentation.py` executes it.

## Mixed rows

Measured on the shipped code, same conditions (`pr1/confirm/accept/oracle_mixed.py`). The
phrase set is every English and Spanish pattern of 2 to 5 words that changes language, 208
phrases, against a hand-written layout and a phoneme oracle that reads the lexicon files
directly.

| check | result |
|---|---|
| Orthography against the layout oracle | 208 of 208 |
| Phonology against the word-by-word oracle | 208 of 208 |
| Single-language controls, both modalities | 64 of 64 |
| A row alone against the same row in a slice, csv and pkl | 208 of 208 each |
| Decode, then encode again | 208 of 208 |
| Planted: one language per row | flagged on 208 of 208 |
| Planted: the switch token before the space | flagged on 208 of 208 by the layout oracle |

The research prototype flagged 177 of 208 for the one-language plant, and the shipped code
flags all 208. The prototype fell back to English for a word missing from its language,
which hid the plant on Spanish-first phrases. With that fallback put back, the shipped
code flags 179 of 208, as predicted from the lexicon files before the run. The rest of the
gap to 177 is the sample, drawn with another seed.

A model uses the switch. With `d_model` 64, `d_embedding` 8 (the memory rows the decoder
reads), batch 32, 5000 steps and seeds 0, 1 and 2, scored by greedy generation on held-out
rows (`pr1/e2e/run.py`, median [min, max]):

| task | per-word languages | first code only | every row `--` |
|---|---|---|---|
| `o2p` on 600 rows, pairs of words spelled alike in both languages | 1.000 [1.000, 1.000] | 0.202 [0.200, 0.238] | 0.038 [0.033, 0.045] |
| `p2o` on 300 rows, spelling with switches out | 1.000 [1.000, 1.000]; 735 of 735 switches in the right place with the right id | | |

With the first code only, word 1 stays at 1.000 and word 2 falls to 0.202. An untrained
model scores 0.000 in every arm.

Those three-seed runs predate the load-time checks in the next section. Seed 0 was run
again on the source this record describes. All 404 recorded values equal the earlier
run's, floats included (`pr1/confirm/e2e/compare_with_earlier.py`). Seeds 1 and 2 were not
run again. The `--` arm read a file tagged `--`, which the load rejects, so the repeat
takes its inputs from a tokenizer call with no language. Those inputs equal the `--`
layout on 5640 of 5640 training rows and 600 of 600 test rows
(`pr1/confirm/e2e/run_adapted.py`). The data files of the other arms give the same tensors
with the checks in place: every row equals an answer key built from the raw lexicon files,
76 of 76 checks (`pr1/confirm/e2e/checks_adapted.py`).

The scores are ceiling scores on a small closed vocabulary. They show the mechanism works,
not how it holds up on new words.

## What is checked, and when

A language is an argument error, so it raises. A missing pronunciation is a data condition,
so `encode` returns `None`. The dataset draws the same line.

- **At load**, every `language` cell is checked and a bad one raises `ValueError` naming
  the row: a blank or missing cell, `--`, an unsupported code, a wrong count. `--` is for a
  direct tokenizer call; a file never holds it. Of 121 constructed inputs, 94 behave as
  the design says and none against it. The design is silent on the other 27, and their
  outcomes are recorded (`pr1/confirm/accept/boundaries.py`). An unsupported code gives one
  message from 10 entry points, and a wrong count one message from 5.
- **A comma between two codes** makes three fields under a two-field header. pandas 2.3.3
  reads such a file without complaint when every row has the extra field. It takes the
  first field as an index, so at `9e7d9f2` the codes loaded as the words
  (`pr1/confirm/findings/rc/p09_index_trap.py`). The local csv read therefore raises when
  pandas hands back anything but a `RangeIndex`, the plain row count 0, 1, 2, which is the
  mark of the extra field. Six ordinary csv shapes load: header only, with and without a
  written index column, an extra named column, CRLF line ends, quoted cells
  (`pr1/confirm/new/n2_csv_shapes.py`).
- **The check covers the local csv read only.** A frame from a `gs://` client may be
  filtered, sampled or sorted, which gaps its index without adding a field. Applied to
  every frame, the check refused 4 of 4 such frames that loaded at `9e7d9f2`
  (`pr1/confirm/new/n1_client_frame_index.py`), so it sits where pandas builds the index.
  All 4 load on the shipped code (`pr1/final/n1_client_frame_index.out`). A comma typo in
  a file read through a client is not caught.
- **Languages are a string, a list or a tuple.** Codes are matched to words by position,
  and a set gave a word order that followed the hash seed. A set, a dict, a numpy array
  and a pandas Series each raise `ValueError`, the same under 6 hash seeds
  (`pr1/confirm/findings/rc/p07_input_types.py`).
- **Pronunciations are not checked at load.** A word with none raises
  `RuntimeError: Failed to encode word(s)` when its row is read, and the message lists
  every row of the slice. The first row is the exception: the load encodes it once and
  leaves it out with a warning. That asymmetry is the baseline's. Removing the English
  fallback widens who meets it: at `9e7d9f2` a Spanish row holding a word that only
  English has was encoded, with English phonemes (`pr1/confirm/e2e/missing_word.py`).
- **A header other than `language`** is not noticed. A file with no `language` column is
  English throughout, which the 7300-row fixture depends on, so a column spelled
  `languages` loads as English.
- **A dropped code cannot be caught.** `ES EN` mistyped as `ES` on a two-word row is legal,
  because one code means every word.

## Accuracy is reported with and without switches

A switch is a scored token, so one missed switch costs a letter and fails its row.
Orthographic accuracy is therefore reported as two pairs.

- `letter_wise_accuracy` and `word_wise_accuracy` count every scored token, switches
  included. They score exactly the positions the loss trains and are for diagnosing the
  model.
- `letter_wise_accuracy_without_switches` and `word_wise_accuracy_without_switches` leave
  out the positions whose target is a language token. They measure spelling alone and are
  the figures for research claims. On a row with no switch the two pairs are equal.

| check | result | script |
|---|---|---|
| Rows with no switch: the two pairs are equal | 120 of 120 values, 78 of 78 values, and 107 of 107 held-out rows | `diff_training.py`, `diff_single.py`, `pr1/confirm/e2e/metrics_check.py` |
| A prediction wrong only at one switch, against the analytic value | with switches (n-1)/n and 0; without switches 1 and 1; 208 of 208 phrases | `oracle_mixed.py` |
| A prediction wrong only at one letter | both pairs drop, 208 of 208 phrases | same |
| A `p2o` model partway through training, 193 held-out rows with a switch | 18 rows at 600 steps and 15 at 1200 have every letter right and a switch wrong. Each scores 0.0 with switches and 1.0 without | `metrics_check.py`, seed 0 |
| `evaluate` against figures recomputed from the logits, 4 figures at 4 points in training | 0 of 300 rows differ, 16 times of 16 | same |

This amends one sentence of 0004, which kept the metric on the loss's positions because
splitting them had hidden a defect. The first pair still honours that. The second departs
from it on purpose, by positions that are named, and only beside the first.

Telling a switch from a letter takes the language ids, which `VocabSpec.from_tokenizer`
records as `orth_language_ids`. A spec that does not name them, written by hand or
unpickled from a checkpoint saved at `9e7d9f2`, reports the first pair only. The second is
absent, not estimated.

## Consequences

**A mixed row is longer by one token per switch**, and the 30-slot position table counts
them. A row 30 tokens wide trains and the same row with one switch, 31 wide, raises
(`pr1/confirm/e2e/awkward.py`). In the research sample 146 of 208 mixed phrases exceed the
table, 138 of them with no switch at all (`wl_proto.md`).

**The tests pin the behaviour.** The suite is 710 passed and 6 skipped, against 565 and 6
at `9e7d9f2` (`pr1/confirm/02_suite.out`). 130 defects were planted one at a time in fresh
copies of the tree, each followed by the full suite. 127 failed at least one test. The
other 3 did not match the source, which had moved since they were written, and the same
three defects, written again by hand, are among the 127. Four harness controls behave: two
no-op edits fail nothing and two import canaries stop the run
(`pr1/confirm/mutation/run_confirm.py`, `all_plants.txt`).

`decode_with_languages` cannot see which side of a space a switch sits on. It accepts all
208 rows of the switch-before-space plant, so only the literal layouts in
`tests/domain/tokenizer/test_per_word_languages.py` pin that.

`decode` is the text half of `decode_with_languages`, one implementation for both. It
returns the baseline's text on 25000 of 25000 rows
(`pr1/confirm/findings/fx/decode_diff.py`). A prototype of that change decoded 64 rows of
24 ids in 420 microseconds median against 184
(`pr1/review_simplicity/probes/p_s1_decode.py`). It was not timed again on this source.
Nothing in the training pipeline or the model calls `decode`: a search of `bridge/` for
`.decode(` finds it only inside the tokenizers.

A wrong seed mislabels in silence. In the research runs a decoder seeded with the wrong
language spelled 300 of 300 rows correctly and labelled all 300 wrong (`wl_audit.md`).

For a list of rows, `languages=["ES", "EN"]` is one entry per row. For one row it is one
code per word. A list of two two-word rows given that value is read per row.

A training event names its rows by string, and a string does not identify a row.

Not covered here, and filed: accented Spanish letters fall outside the character
vocabulary, so `qué` decodes as `qu` (#237); seven Spanish phonemes are missing from the
table (#238); phonology has no language token, so `o2p` and `p2p` cannot be told which
language to produce (#239).

Not measured:

- CUDA, for any number in this record.
- Seeds 1 and 2 of the 5000-step runs on the source this record describes.
- A trained model on new words, on rows of more than three words, on a third language or
  on real text.
- How often a real Spanish file holds a word its lexicon lacks, which is what decides how
  many rows the missing fallback stops.

Scripts and outputs are under `pr1/` and `wf_*/` in the evidence folder that
`docs/specs/input-target-pairs.md` names. Those under `pr1/confirm/` were run on
library source digest `83006f726aac4408`, named in their headers. One change came after
them: the comma check moved from every frame to the local csv read. The five acceptance
probes were then run again unchanged and each passed with the same counts, and the suite
gave 711 passed and 6 skipped (`pr1/final/`).
