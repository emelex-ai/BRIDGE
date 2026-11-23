# Spec: input and target are different things

Status: decisions recorded 2026-11-22. Sections 1 and 2 are implemented. Section 1 (PR 1, language per word) is described as shipped by decision record 0015, `docs/decisions/0015-language-belongs-to-each-word.md`. Section 2 (PR 2, input and target as types) is described as shipped by decision record 0016, `docs/decisions/0016-a-training-example-is-an-input-and-a-target.md`. The numbers in sections 1 and 2 are the prototypes'; the records carry the ones measured on the shipped code. Section 3 (PR 3) is not implemented.

## The answer

Today every training example is one string that the model reads and then reproduces. This
spec makes an example two strings, an input and a target, each with a language per word. They
may be equal, and repetition is then just a dataset whose two columns match.

It is four changes, shipped as three pull requests:

| PR | change | previous behaviour |
|---|---|---|
| 1 | Each word carries its own language. Fixes the Spanish phrase bug and allows mixed-language rows. | Bitwise identical for every single-language row that was encoded correctly (measured) |
| 2 | Input and target become separate types through tokenizer, dataset, pipeline and generation | Bitwise identical when target equals input (measured on a prototype) |
| 3 | Positions are computed for any length. The 30-slot tables go. | Not preserved. Every model output changes and old checkpoints stop loading |

PR 3 breaks your third constraint, "strictly a feature addition". No position scheme that
lifts the cap can keep it. Section 3 says what the break buys.

Every number below was measured at commit `9e7d9f2`, CPU, float32, torch 2.12.0+cu130, one
thread. Scripts and outputs are listed under Evidence.

## 1. Language per word (PR 1)

**What exists.** A row's language reaches the tokenizers as `language_map`, a dict keyed by
string. `CharacterTokenizer` (characters to ids) looks up the whole row string.
`PhonemeTokenizer` (words to phoneme ids) looks up each word of the row separately. A
phrase's words are not keys, so they fall back to English.

**What that causes.** Ten defects reproduced, all from that one mismatch. The three that
matter most:

| defect | observed |
|---|---|
| A phrase tagged Spanish gets English phonemes | "no me" tagged ES encodes as `[86, 69, 33, 89, 68, 32, 87]`, the English rows. Spanish is `[86, 69, 22, 89, 68, 27, 87]` |
| A row's phonemes depend on its batch neighbours | "no me" tagged EN, beside a row "no" tagged ES, gets a Spanish "no". A shuffled epoch changes the same row |
| One string under two languages collapses in a slice | rows "no"/EN and "no"/ES both come out Spanish |

No test catches any of them. No test compares a phoneme id against a language.

**The change.** Each row carries one language code per word. A single code means every
word. Think of a teacher's sentence, mostly English with one Spanish word:

```
the word for cat is gato        EN EN EN EN EN ES
[EN] [BOS] t h e _ w o r d _ f o r _ c a t _ i s _ [ES] g a t o [EOS]
```

- **Phonology.** Each word is looked up in its own language, by its position in the row, never by string.
- **Orthography.** The opening token is the first word's language. A language token appears again only where the language changes, directly before that word's first character.
- **Targets.** A mixed-language target has the same layout, so the decoder learns to emit the switch itself.
- **Data file.** The `language` cell is `EN` or `EN EN EN EN EN ES`. A pkl file may hold a real list.
- **Decoding.** A new `decode_with_languages` returns the text and each word's language.

The language tokens are already ordinary entries in the orthographic vocabulary (ids 6, 7
and 8 of 109), so the model does not change.

**Rules the layout needs.** Each was chosen in the prototype and checked.

| point | rule |
|---|---|
| What a word is | One shared function, `\S+`, used by both tokenizers. It agrees with `str.split()` on every Unicode code point |
| Whitespace | Characters are kept exactly as written. This is what keeps single-language rows identical |
| Code count | Must be 1 or equal the word count. Otherwise `ValueError` naming the row, at load |
| Unsupported code | `ValueError`, the same one everywhere, at load |
| No language | `--` for the whole row, with English phonemes. `--` beside real codes raises. The dataset never produces it: a blank `language` cell raises at load |
| Word missing from its language's lexicon | Treated like any missing pronunciation. No fallback to English |
| Stored form | A tuple per row, so rows stay hashable and sortable |

Removed: `language_map`, `_resolve_language_map`, `_word_languages`, `strict_conflicts`,
string indexing (`ds["cat"]`) and `get_encoding`. They exist to look a language up by
string, which pairs make ambiguous.

**Measured: the encoding is right.** Prototype against today's code and independent oracles.

| check | result |
|---|---|
| Single-language rows against today's code, 18 groups: words, phrases, batches, odd whitespace, capitals | 0 failures; 3000 of 3000 English words, 3000 of 3000 Spanish, 7300 of 7300 fixture rows |
| The ten defects | all fixed or removed |
| Mixed rows against a hand-written layout and a word-by-word phoneme oracle | 208 of 208, covering every English and Spanish pattern of 2 to 5 words |
| A row alone against the same row in a batch | 1248 of 1248 |
| Decode then re-encode | 208 of 208 |
| The shipped suite run through the prototype | 564 passed, 1 failed. The failure counts calls to two inner methods, not behaviour. The golden master passes 24 of 24 |
| Independent audit: own oracle, fresh phrases, fuzz | 100 of 100 and 3000 of 3000 |

Planted defects were caught each time, including one-language-per-row (177 of 208 flagged,
exactly the 177 predicted).

**Measured: a model uses the switch.** Unmodified model, `d_model` 64, 8 memory rows, 5000
steps, three seeds, scored on held-out phrases by greedy generation. Cells are median
[min, max].

| question | test | result |
|---|---|---|
| Does the encoder read it? | `o2p` on pairs of words spelled alike in both languages, so the tokens are the only cue to pronunciation | 1.000 [1.000, 1.000] |
| Control: opening token only, as today | same data | 0.253 [0.210, 0.262]. Word 1 stays perfect; word 2 collapses |
| Control: every language token blanked | same data | 0.053 [0.053, 0.063] |
| Does the decoder emit it? | `p2o`, phonemes in, spelling with switches out | 1.000; 735 of 735 switches at the right place with the right id |
| Does the seed steer the output? | `o2o`, two targets per input, differing only by seed | 1.000; the same seed twice never flips |

The audit retrained seven of these models and got the same numbers bit for bit. It then
edited inputs by hand: with the switch deleted, word 2 reverted to the opening language on
300 of 300 rows. Across 6000 orthographic generations, 3525 with a switch mid-sequence,
nothing broke.

These are ceiling scores on a small closed vocabulary. They show the mechanism works. They
do not show how it holds up on new words or real text.

**What the research turned up that needs handling.**

- **Accuracy metrics count a switch as a letter.** One missed switch fails the whole row. Two figures are reported. The first counts every token, switches included, and is for diagnosing the model. The second counts letters only, and is the one for research claims.
- **A wrong seed mislabels silently.** Seeded with the wrong language, the decoder spelled 300 of 300 rows correctly and labelled all 300 wrong.
- **A blank `language` cell loads as "unspecified"** today and gets English phonemes. The dataset raises at load, naming the row.
- **A dropped code on a two-word row cannot be caught.** `ES EN` mistyped as `ES` is legal, because one code means every word. No file format fixes this; JSON was tested and is no safer.
- **Mixed phrases outgrow the 30-slot cap.** 146 of 208 sample phrases exceed it, though 138 would with no switch at all. Real use of this feature waits on PR 3.

**Two Spanish defects that exist today, outside this change.**

- **1344 of 7095 Spanish words lose letters.** The character vocabulary is ASCII, so `qué` decodes as `qu`.
- **2639 of 7095 Spanish words contain a phoneme the table lacks**, and encode to the unknown row. English has 2 such words in 123,472.

Both limit any Spanish result, including translation. Neither is fixed here. They are filed
as issues #237 (letters) and #238 (phonemes).

**One limit this change leaves.** Phonology has no language token, so `o2p` and `p2p`
cannot be told which language to produce. Translation into phonology works only where the
input implies the output language. Filed as issue #239.

## 2. Input and target as types (PR 2)

**What exists.** `EncodingComponent` (the five tensors for one modality of one string) holds
both what the encoder reads and what the decoder is taught. That only makes sense when the
two are the same string. `Model.forward` itself needs no change: the decoder reads a
fixed-size memory, so the two lengths are already independent.

**The change.** Every tensor gets exactly one reader, and each type's name says which side
of the example it is: the input side, which the encoder reads, or the target side, which
the decoder is fed and scored against. Think of a flashcard: the front is
what you read, the back is what you must say. Today's type is a card with the same text
printed on both sides.

```python
class Tokens:        ids, pad_mask                 # what an encoder reads
class Target:        ids, pad_mask, labels         # what a decoder is fed, and scored against
class BridgeInput:   orth: Tokens | None, phon: Tokens | None
class BridgeTarget:  orth: Target, phon: Target
class BridgeBatch:   input: BridgeInput, target: BridgeTarget
```

`EncodingComponent` and `BridgeEncoding` are deleted. Three designs were prototyped and
judged. This one won on "distinct foundationally". The cheapest alternative keeps the two
old types and refills them, with zero new types and far fewer test edits, but nothing in it
says which half is which.

| piece | change |
|---|---|
| Tokenizer | `encode_input(texts, languages)`, `encode_target(texts, languages)`, `orth_start(languages)`. `languages` is one entry per row, each a code or a list of codes |
| Orthographic labels | Stored on the target, like phonological ones. The shift that decision 0004 defines moves from the loss into the tokenizer, with equal values |
| Dataset columns | `word_raw`, `language`, `target_raw`, `target_language`. `target_raw` is required; a file without it raises with a message naming the fix. A missing `target_language` copies `language` only when that is a single code; reusing a per-word list was measured silently wrong when the word counts happen to match |
| Dataset storage | One list of rows replaces the parallel `words` and `languages` lists, so a shuffle cannot misalign them |
| `dataset[i]` | Returns a `BridgeBatch`. Integer and slice only |
| Encoding memo | Deleted. It ran 0 times in 6 training steps |
| Pipeline | `forward(batch)`, `compute_loss(logits, target)`, `compute_metrics(logits, target)`. `single_step` and `evaluate` keep their signatures |
| Events | `"input"` and `"target"` replace `"word"` |
| Generation | `Model.generate(inputs, pathway, *, max_new_tokens, orth_start=None, ...)` reads only the input. The cap arrives with PR 3 |

**Naming the output language.** A pathway that writes orthography requires `orth_start`,
the `[LANG, BOS]` seed, built by `tokenizer.orth_start(["ES", ...])`. Omitting it raises.
It names the language of the first output word; later switches come from the decoder.
Today the seed is silently the input's language. Measured on a task whose target opens in
the other language half the time, today's `generate` scores exactly 0.500.

**Optional input modalities.** A nonword has no phonemes, and a phonology-only input has no
spelling. Today both carry fake placeholder tensors. With `None` instead, a pathway asked to
read a missing modality fails at the boundary with a message naming it. That closes the
second failure listed in issue #233.

**Measured on the prototype.** With target equal to input, against today's dataset and
pipeline on 48 rows: logits, loss, metrics, 6 training steps, 169 of 169 parameters and all
generation output are bitwise equal on 5 of 5 pathways. A mixed-language pair batch with
different input and target lengths trains, evaluates and generates on 5 of 5.

**A trap the tests must cover.** Repetition tests cannot see input and target wired the
wrong way round, because both sides hold equal tensors. A planted swap passed all of them.
Two permanent tests catch it:

- **Read-set test.** Roll one tensor along the batch. The loss must move exactly when the design says that tensor is read.
- **Hand-wired reference.** `forward(batch)` must equal `Model.forward` called with tensors picked by hand.

## 3. Positions for any length (PR 3)

**What exists.** Two learned tables of 30 rows, one per modality. A sequence of 31 tokens
raises. The same 30 bounds the generation loop.

**The bake-off.** Six schemes on one harness, three tasks, trained on phrases of 5 to 19
tokens, with `d_model` 64 and 8 memory rows. Cells below are median [min, max] over seeds
0, 3, 4 and 5; two more seeds agree. The harness's copy of today's scheme was checked
bitwise against the real model. Its planted defects were caught.

| scheme | at trained lengths | reads a longer input | step time |
|---|---|---|---|
| today's 30-row table | baseline | raises past 30 | 1.00 |
| wider table | no detectable gain | 0.00 | 1.00 |
| sinusoidal (fixed waves, about 10 lines) | tied | 0.00 | 1.00 |
| word and letter counters | slightly behind on one task | one extra word only | 1.01 |
| rotary, rotary mixer | tied | 0.47 [0.28, 0.94] | 1.05 |
| **rotary, stock mixer** | tied | **0.84 [0.36, 0.99]** | 1.03 |

"Reads a longer input" is exact match on a task that hears a 23 to 30 token phrase and
spells its last word, having trained on at most 19 tokens. Rotary means position enters as a
rotation inside self-attention, so the model sees distances, not slot numbers.

**Recommendation: rotary in the two encoders and two decoders, mixer left alone.** A second
run trained to 47 tokens and tested at 51 to 61. It scored 0.93 where the table and the
waves scored under 0.01. That run has two seeds, so its spread is thin.

**What it costs.**

- About 130 lines of owned attention blocks replace the 66-line wrappers in `bridge/domain/model/layers.py`. Rotary cannot pass through PyTorch's stock layers; that trap was re-measured.
- 3.4% per training step. The gap between two identical models is under 1%.
- Parameters go from 169 tensors to 167. Old checkpoints fail to load and would be meaningless if they did.
- `max_orth_seq_len`, `max_phon_seq_len`, `_positions` and the length check are removed.
- `generate` needs a length cap from the caller, since the table no longer supplies one.
- The `d_embedding` default goes from 1 to 2. It lands here because it changes every default model's output, and this PR already does.

**The length cap.** `generate` takes `max_new_tokens`, keyword-only and required. It counts
tokens produced after the seed. A sequence stops at `[EOS]` or at the cap, whichever comes
first, and one that hits the cap ends without `[EOS]`, which is how a caller sees it was cut.

What the main generation routines do, read from their documentation on 2026-11-22:

| framework | the cap | when the caller omits it |
|---|---|---|
| Hugging Face `generate` | `max_new_tokens`, new tokens only | 20 tokens in total, with a warning. The docs say to always set it |
| fairseq | `max_len_a * source_length + max_len_b` | 0 and 200, so a flat 200 |
| llama.cpp | `n_predict` | -1, no limit |
| vLLM chat endpoint | `max_tokens` | whatever is left of the model's context window (from a secondary source) |
| Anthropic Messages API | `max_tokens` | rejected; the field is required |

All five have a cap and stop at end-of-sequence. They split on omission. The defaults that
work are backed by a fixed context window, which this model stops having. The one that is
not, Hugging Face's 20, cuts output silently and its own docs warn against relying on it.
So the name and the counting follow Hugging Face, and the missing default follows Anthropic.

**Two limits that hold for every scheme.** The spec must not promise otherwise.

- **Nothing repeats a phrase longer than it trained on.** Exact match is 0.000 for all six schemes. Training data must cover the lengths you will use.
- **`d_embedding` (the number of memory rows the decoder reads) defaults to 1, and 1 cannot carry a phrase.** Repeating 33 to 47 trained tokens scored 0.00 at 1 row, 0.49 to 0.67 at 8, and 0.95 to 0.97 at 32. This was measured with the sinusoidal scheme only. The new default of 2 was not run, and nothing measured says 2 rows carry a phrase. Runs on phrases should set it explicitly.

## 4. Tests, fixtures, docs, figure

Baseline: 565 passed, 6 skipped, 360 test functions.

| PR | existing tests to edit or delete | new tests |
|---|---|---|
| 1 | 24 functions, plus about 6 that read a row's language as one string | exact layout of a mixed row; "no me"/ES regression; a row encodes the same alone and in any batch; decode and re-encode identity; count mismatch and blank cell at load; a metric case where only the switch is wrong |
| 2 | about 560 lines in 26 of 49 files, mostly field renames | read-set; hand-wired reference; generation reads only the input and `orth_start`; boundary errors for missing modalities |
| 3 | 27 functions | shifting every position by k changes nothing (analytic); step-by-step generation equals one full decode; padded batch equals single items; a 60-token sequence runs |

**Golden master.** `tests/fixtures/phon_baseline.pt` (the recorded fingerprint of the
phonological path) is not regenerated. PRs 1 and 2 leave all 24 of its test nodes passing.
PR 3 moves every model-derived float, so 12 nodes stay exact (tokenizer and vocabulary) and
12 reduce to shapes, following the pattern set for issue #228. New behaviour is pinned by
the analytic tests above. `tests/fixtures/capture_phon_baseline.py` is left alone, since it
only ever runs on old commits.

**Figure.** `BRIDGE-Models.drawio` needs a redraw for PR 2: the input and output boxes show
the same three words, and one embedding cube feeds both encoder and decoder. For PR 3 the
label "Embedding + Position" becomes false.

**Documents.** `docs/architecture.md` (shape section, flow block, layout, the checkpoint
sentence), `README.md` (multilingual example, dataset format), both rule files, and three
new decision records:

- **0015** Language belongs to each word of a row, by position, and orthography marks each change. Amends 0008 and one sentence of 0004.
- **0016** A training example is an input and a target. Amends 0004 and 0013.
- **0017** Positions are computed for any length. Supersedes parts of 0002 and 0008.

## 5. Acceptance, per PR

Each PR is done when the suite, `mypy` and `ruff` pass, the documents above are true, and:

- **PR 1.** A differential against a pristine worktree shows every single-language row that was encoded correctly bitwise equal. The differential is first shown to catch a planted language change. The 208-phrase layout and phoneme oracles pass on the shipped code, with the one-language-per-row plant caught.
- **PR 2.** A 200-row, 12-step differential against the PR 1 commit, target equal to input, is bitwise on all five pathways, with a same-commit control row. The planted swap of input and target fails the two new tests.
- **PR 3.** A `d_embedding` sweep over 1, 2 and 8 on the shipped code is recorded in decision record 0017. The five correctness probes from the bake-off pass on the shipped code, each with its planted defect caught. The bake-off's headline cell reproduces on the shipped code.

## 6. Decisions

Decided by Nathan on 2026-11-22. The sections above already say the same.

| # | question | decision |
|---|---|---|
| 1 | Five types named for their side, or today's two types reused? | Five types |
| 2 | Is the `target_raw` column required? | Yes, with an error naming the fix. Cost: every existing CSV and up to 73 test functions gain a column |
| 3 | A word missing from its tagged language falls back to English. Keep? | No. It is treated like any missing pronunciation |
| 4 | No language on a direct tokenizer call gives `--` with English phonemes. Keep? | Keep |
| 5 | How should accuracy treat a switch token? | Report both: every token, for diagnosing the model; letters only, for research |
| 6 | Phonology has no language token | Logged as issue #239. Not fixed here |
| 7 | Rotary with stock mixer, or sinusoidal? | Rotary |
| 8 | The generation length cap | `max_new_tokens`, keyword-only, required. See section 3 for what other frameworks do |
| 9 | Old checkpoints stop loading after PR 3 | Accepted, with no migration path |
| 10 | The `d_embedding` default | Raised to 2, in PR 3 |
| 11 | The two Spanish defects | Logged as issues #237 and #238. Fix before any Spanish result is reported |
| 12 | A blank `language` cell | Raises at load |

**Decision 1, how input and target are typed.** Five small types named for their side, as
section 2 describes. `BridgeInput` holds what the encoder reads. `BridgeTarget` holds what
the decoder is fed and scored against. Passing one where the other belongs is a type error.
The alternative kept today's two types and passed two `BridgeEncoding` objects per example;
a planted swap of the two passed every repetition test.

## 7. Not measured

- CUDA, for anything in this spec. Numerics and step time are CPU only.
- `p2p` and `op2op` under the new position scheme, beyond consistency checks.
- Real corpora, punctuation, and words that are not one phoneme per letter. The bake-off pool is 204 short words.
- `d_model` above 64, more than 3500 steps, other learning rates.
- How often the English fallback in decision 3 fires on a real Spanish corpus.
- The switch token with new words, more than three words, a third language, or text near the length cap.
- The switch token under rotary positions. The bake-off had no mid-sequence language token.
- Accented Spanish words in any mixed-row check. They were excluded by construction.
- The `d_embedding` result under rotary.
- The winning design with required `target_raw`, and with a runtime check that an input is never passed where a target belongs. Neither has a prototype.
- `d_embedding` at 2. Only 1, 8 and 32 were run.
- The phonology language limit in issue #239. It is read from the code, not measured on a trained model.

## Evidence

Scripts and their outputs, without model checkpoints, are kept outside the repo in the
author's evidence folder, `evidence/input-target-spec/`. Paths in this spec and in the
decision records are relative to it.

| claim | script and output, relative to that folder |
|---|---|
| language defects | `wf_language/defects.py`, `defects.out` |
| per-word prototype | `wf_wordlang/proto/wordlang.py` |
| single-language equivalence, defects, mixed rows | `wf_wordlang/proto/b_equiv.out`, `c_defects.out`, `d_mixed.out` |
| word split, Spanish lexicon defects | `wf_wordlang/proto/f_wordsplit.out`, `h_unk_phoneme.out` |
| model uses the switch | `wf_wordlang/learn/run.py`, `summary.out` |
| audit, hand-edited inputs | `wf_wordlang/audit/learn/e1_extra_summary.out`, `e2_extra_full_0.out` |
| dataset, file formats, suite through the prototype | `wf_wordlang/edges/e1_dataset.out`, `e2_formats.out`, `run_proto.txt` |
| winning design, repetition bitwise | `wf_design_first-principles/fp_evidence.py`, `fp_evidence.out` |
| input and target swapped, missing-modality errors | `wf_design_judge/hunt.out`, `graft_optional.out` |
| read-set and reference oracles | `wf_design_two-encodings/pair_run.py`, `pair_run.out` |
| position bake-off, six seeds | `wf_positions/final/table_pooled_6seeds.md` |
| trained-long control, `d_embedding` | `wf_positions/audit/trainlong/table_trainlong.md` |
| stock-layer trap | `wf_positions/probes_rope/trap_stock_layers.out` |
| test and document inventory | `res_inventory.md`, `wf_inventory/` |
| full reports from each agent | `res_language.md`, `res_design.md`, `res_positions.md`, `wl_proto.md`, `wl_learn.md`, `wl_edges.md`, `wl_audit.md` |
