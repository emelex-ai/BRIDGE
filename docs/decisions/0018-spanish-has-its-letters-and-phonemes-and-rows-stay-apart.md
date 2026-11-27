# 0018. Spanish has its letters and phonemes, and rows are coded to stay apart

Status: Accepted
Date: 2026-11-27
Amends [0002](0002-phoneme-row-ids-replace-ragged-feature-lists.md), whose table had 91
rows. Closes issues #237 and #238.

## Context

Two defects limited every Spanish result. Both were measured at `3d28ee9`, on CPU.

| defect | observed |
|---|---|
| The character vocabulary was ASCII | 1344 of 7095 Spanish lexicon words lost a letter. "qué" decoded as "qu" |
| Seven Spanish phonemes had no row in `phonreps.csv` | 2639 of 7095 Spanish words encoded a phoneme as `[UNK]`. "niño" lost its "ñ" sound |
| Two English entries lacked a stress digit | "CATTAIL" and "CATTAILS" held a bare `EY`, which has no row |

A third matter came up while the table was open. A decoded position is scored as the row
nearest to it. A row one feature from another is then one wrong feature from being read
as the other phoneme: "ban" as "bang", "fin" as "thin". At `3d28ee9`, 14 pairs of
phonemes the lexicons use sat one feature apart, stress set aside. Some of those pairs
are real. `B` and `P` differ in voicing and in nothing else. Others came from how a sound
had been written down, and several of those codings are ones phoneticians disagree about.

## Decision

**Sixteen characters join the orthographic vocabulary, after everything it held.** They
are `á é í ó ú ü ñ`, the same seven as capitals, `¿` and `¡`, at ids 109 to 124. The list
is written out in the code as `SPANISH_CHARACTERS` and is never read from a lexicon file,
so an id means the same thing whatever lexicon is on disk. No existing id moves.

**A row is composed to Unicode NFC before either tokenizer reads it.** An accented letter
can arrive as one code point or as a letter and a combining mark. Both now give the same
ids and the same pronunciation. NFKC was not used: it changes word counts and rewrites
characters such as the "fi" ligature.

**Seven phonemes join the table, after the special tokens.** The special tokens stay at
rows 86 to 90 and the new rows take 91 to 97, so no existing row id moves. Each row uses
only the 31 features the table had.

| symbol | row | sound | as in | features |
|---|---|---|---|---|
| `X` | 91 | [x], voiceless velar fricative | "mejor" | velar, fricative |
| `_BB_` | 92 | [β], voiced bilabial fricative | "sabe" | labial, fricative, voice |
| `_GG_` | 93 | [ɣ], voiced velar fricative | "algo" | velar, fricative, voice |
| `_JJ_` | 94 | [ʎ], palatal lateral | "calle" | palatal, liquid, voice |
| `_LL_` | 95 | [ʝ], voiced palatal fricative | "yo" | palatal, fricative, glide, voice |
| `_NN_` | 96 | [ɲ], palatal nasal | "niño" | palatal, nasal, voice |
| `_RR_` | 97 | [r], alveolar trill | "perro" | alveolar, liquid, rhotic, tap, voice |

Two of the lexicon's names mislead. `_JJ_` is the sound spelt "ll" and `_LL_` is the
sound spelt "y". The rows follow the words, not the names: all 307 words holding `_JJ_`
are spelt with "ll", and none of the 98 holding `_LL_` is.

**The two English entries are corrected in the lexicon.** Both now hold `EY2`, which is
what 14 of the 15 other "-tail" compounds with first-part stress carry. A table row or a
tokenizer rule for two typos would have added a mechanism for a data defect.

**Where a coding is debated, the table takes the one that leaves rows farther apart.**
The feature table follows Harm and Seidenberg (1999, 2004), with stress added. How to
write several of these sounds as binary features is a matter phoneticians still argue,
and this record does not claim to settle any of it. It makes a narrower choice, for a
narrower reason. This model scores a position by its nearest row, so the distance between
two rows decides which errors read as another phoneme. Where sources disagree, or where a
sound can fairly be written two ways, the table takes the coding that best separates the
phoneme vectors. A pair stays one feature apart only when the two sounds differ in one
thing.

Five codings were set under that rule or corrected.

| rows | was | is | decided by |
|---|---|---|---|
| `AE0`, `AE1`, `AE2` | front and back | front | a correction: the vowel of "cat" is front |
| `W` | voiceless | voiced | a correction: `W` is the voiced glide |
| `WH` | voiced | voiceless | the same correction. No word of either lexicon uses `WH` |
| `NG` | alveolar and velar | velar | accuracy and separation agree. The IPA chart has it velar alone, and it now sits two features from `N`, as `M` does |
| `CH`, `JH` | affricate | stop, fricative and affricate | the rule. An affricate is a stop released as a fricative, so both are on. `affricate` stays on beside them and holds `CH` two features from `SH` |

The trill `_RR_` was set the same way. `L` made rhotic would sit one feature from `L`.
A trill is a run of taps, so the row carries `tap` as well, and sits two features from
`L` and two from `DX`, the tap of "pero".

The codings left alone are the ones where no edit changes which rows are neighbours:
`EH` with `tense` and two heights, and `EY` with two heights. `IX` is coded as a back
vowel and no word uses it.

## What a row being one feature apart looks like

Among the phonemes the lexicons use, stress set aside:

| table | pairs one feature apart |
|---|---|
| `3d28ee9` | 14 |
| with the seven Spanish rows as first written, `_RR_` as `L` made rhotic | 19 |
| this record's table | 17 |

The 17 are nine voicing pairs, such as `B`/`P` and the Spanish `X`/`_GG_`; two tense
pairs, `IH`/`IY` and `UH`/`UW`; two labial pairs, `F`/`TH` and `V`/`DH`; `AO`/`OY`; and
three more that the Spanish rows add.

| pair | differs in | why it stands |
|---|---|---|
| `V`/`_BB_` | dental | bilabial against labiodental is one difference of place. No lexicon holds both |
| `Y`/`_LL_` | fricative | [ʝ] is the glide of "bien" made a fricative, one difference |
| `ZH`/`_LL_` | glide | the plain row for [ʝ] is `ZH`'s, so `glide` is what tells them apart. No lexicon holds both |

Counted over every row of the table, the pairs are 21: the 17, `W`/`WH`, and three that
touch a row no word uses, `DX`/`NX`, `N`/`NX` and `IX`/`UW`.
`tests/core/test_phoneme_table.py` lists all 21, so an edit that puts two rows side by
side, or parts two, fails by name.

## What was measured

All on CPU, torch 2.12.0+cu130, `BRIDGE_DEVICE` unset. Integer ids and strings were
compared exactly. Scripts and outputs are under `spanish-patching/` in the evidence
folder that `docs/specs/input-target-pairs.md` names. Its `letters/` and `phonemes/`
folders hold the runs on each half before the merge, `table-review/` the row edits, and
`merged/` the runs on the final tree, `c58886f`.

| what | reading | script |
|---|---|---|
| Spanish words that fail `decode(encode(w))` | 1344 of 7095 at `3d28ee9`, 0 after. English 0 of 123472 both times | `letters/01_charset.py` |
| Characters outside the old vocabulary, both lexicons | the seven lower-case letters and no others | the same |
| Lexicon keys in a decomposed form | 0 in both files | the same |
| Orthographic ids that moved | 0 of 109 entries. 263264 of 263264 rows that decoded to themselves before are bitwise equal after, 46390938 elements | `letters/03_ids_compare.py` |
| NFC and word boundaries | 0 of 1112064 code points crosses whitespace under NFC. A fuzz of 300000 strings changes 0 word counts; the same fuzz finds 2602 under NFKC | `letters/06_nfc_properties.py` |
| Words holding `[UNK]` in phonology | Spanish 2639 of 7095 and English 2 of 123472 at `3d28ee9`. 0 and 0 on the final tree | `phonemes/measure_lexicon_encoding.py` |
| Phoneme ids of words that held no `[UNK]` before | 127925 of 127925 words bitwise equal on the final tree, 1066293 ids. A planted id was found | `phonemes/compare_encodings.py`, output in `merged/` |
| The row edits and ids | 130566 of 130566 words encode to the same ids before and after the row edits, 1089016 ids | the same |
| The seven rows alone, model at seed 0, 24 steps on five pathways | 1013 of 1013 float tensors and 635 of 635 integer tensors bitwise equal to `3d28ee9`. Measured at `d2da1cf`, before the row edits | `phonemes/compare_model_differential.py` |
| Rows distinct | 98 of 98 | `phonemes/measure_row_distinctness.py` |
| Pairs one feature apart | the table above | `table-review/adjacency.py`, `adjacency_tree.py` |
| A checkpoint saved at `3d28ee9` | refused by `load_model` and by `TrainingConfig.checkpoint_path`: "Tensors of another shape: 3 of 167". 167 of 167 tensors unchanged, `start_epoch` unchanged. A control checkpoint of the new shape loads | `letters/05_checkpoint_load.py`, output in `merged/` |
| Default model | 167 parameter tensors; 627843 scalars, where it had 625779 | `letters/04_checkpoint_save.py` |
| Noise floor of the final tree | 1.262e-07 loss and 4.817e-06 L2, one thread against 6. 1.891e-07 and 5.519e-06, one against 4. A rerun is bitwise equal. A plant of 1e-3 reads 9.971e-05 and 2.171e-04 | `merged/noise_floor.sh`, running record 0017's script unmodified |
| New tests on `3d28ee9` | the lexicon coverage test fails 2, the spelling test fails 9 | `phonemes/new_test_on_origin_main.out`, `letters/08_lexicon_test.main-3d28ee9.out` |
| Planted defects | 25 of 25 caught for the phoneme rows, 17 of 18 for the letters, 13 of 13 for the row edits | `phonemes/plant_defects.py`, `letters/11_planted_defects.py`, `merged/plant_row_edit_defects.py` |

The one plant not caught is the phoneme tokenizer reading rows uncomposed. It changes no
behaviour, because the lexicon lookup composes each word again.

## Consequences

**Every checkpoint saved before this change is refused.** Its three orthographic tensors
have 109 rows where the model has 125, so it fails the shape check of record 0017. There
is no migration.

**The table fingerprint is `7ec53c751f53eb0a`.** It was `4c4a9c388f46bc35`. The phoneme
table is a buffer the model does not save, so a fingerprint that differs gives a warning
at load and not a refusal. That is enough here only because the orthographic shapes
refuse every older checkpoint first.

**The pinned closest-phoneme values moved.** `tests/application/training/test_phon_metrics.py`
scores two committed fixtures against the table. Four of 381 positions that disagreed now
agree: two `N` against `NG`, one `NG` against `N`, and one `SH` against `CH`.

| metric | `3d28ee9` | now |
|---|---|---|
| closest phoneme, L1 and L2 | 0.8530 | 0.8635 |
| closest phoneme, cosine | 0.8504 | 0.8609 |
| the reversed cosine on a scrambled prediction | 0.6237 | 0.5914 |

The script reproduced the old column on the tree before the row edits, then gave the new
one (`table-review/measure_pinned_metrics.py`). The seven Spanish rows alone moved none
of the 381 positions.

**The golden master was not touched.** `tests/fixtures/phon_baseline.pt` has the sha256
it had. Five of its 24 test nodes changed.

| node | how it reads the recording now |
|---|---|
| vocabulary | the recorded `orth_vocab_size` 109 exactly, as the size before `SPANISH_CHARACTERS` |
| feature sets, encoder and decoder side | exactly, with three recorded sets rewritten by hand: `AE1` without `back`, `NG` without `alveolar`, `W` with `voice`. The recording must hold them at four positions a side |
| targets | the same rewrite, on four positions |
| gradient of the phonological embedding | the recording is checked to be the feature count over its own recorded sets, and the model's gradient is compared with that count over the rewritten sets |

With the row edits removed, the four phonological nodes fail. A changed feature, target
bit or gradient element in a copy of the recording fails the node that reads it.

**The figure's feature table lost a row.** `BRIDGE-Models.drawio` shows the targets of
"cat". `AE1` no longer has `back`, so `tests/fixtures/sync_diagram_table.py` rewrote the
table on all five pages.

**A test seed moved.** `tests/domain/model/test_generation_stops_at_eos.py` builds its
model from seed 75. A seed's draws depend on the orthographic vocabulary size, and at 125
the old seed ends no row of `p2o`.

## Limits

- **The phonetics is judgement, not measurement.** The rows follow textbook descriptions
  and the rule above. A phonetician may code any of them differently, and a later record
  can.
- **Three pairs touch a row no word uses.** `NX` and `IX` sit one feature from `DX`, `N`
  and `UW`. A wrong feature at `UW` reads as `IX`. Removing unused rows would move row
  ids, and is not done here.
- **Phonology still has no language token** (issue #239), so `o2p` and `p2p` cannot be
  told which language to write.
- **Other letters still encode as `[UNK]`**: `« » ª º ç à` and every letter outside
  Spanish.
- **Two lexicon oddities are left.** A second variant of "JUANITA" holds a bare `UW`,
  which the tokenizer never reads. `en.json` has one empty key.
- **CUDA numerics are not measured.** The Spanish words encode and all five pathways
  generate on an RTX 5080 (`letters/14_cuda_behaviour.py`). No CUDA float was compared
  with a CPU float, and the noise floor is CPU only.
- **No model was trained to a score.** Nothing here says how the row edits change what a
  trained model gets right.
