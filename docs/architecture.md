# BRIDGE architecture

The current state of the system. This document is living: rewrite it when the structure
changes, and do not narrate the change here. Decisions and their evidence belong in
`docs/decisions/`.

## What the model does

A word has two representations. **Orthography** is its characters. **Phonology** is its
phonemes, where each phoneme is defined by a set of binary phonetic features rather than being
an atom. The model encodes either modality into a shared global representation and decodes
back out to either, which gives five pathways:

| pathway | reads | writes |
|---|---|---|
| `o2p` | orthography | phonology |
| `p2o` | phonology | orthography |
| `o2o` | orthography | orthography |
| `p2p` | phonology | phonology |
| `op2op` | both, cross-attended | both |

Which modalities a pathway reads and writes is declared once, in `PATHWAY_IO`
(`bridge/domain/datamodels/pathways.py`), and every other question, which decoder loop runs, which
loss terms exist, which tensors a training forward needs, which inputs the boundary
validates, is a query against it rather than its own list. All five train and all five
generate. See `docs/decisions/0011`.

`ModelConfig` (`bridge/domain/datamodels/model_config.py`) accepts only sizes for which
that holds. The five layer counts, `d_model`, `nhead` and `d_embedding` are each at least
one. `nhead` divides `d_model` into heads of two or more dimensions. A mistake in a size
fails when the config is built, and the error names the field. See `docs/decisions/0014`
and `docs/decisions/0017`.

| size | default | what it sets |
|---|---|---|
| `d_model` | 64 | the width of every token vector |
| `nhead` | 2 | attention heads; each gets `d_model / nhead` dimensions |
| `d_embedding` | 2 | rows of the global memory, which is all a decoder reads of the input |
| `num_orth_enc_layers`, `num_phon_enc_layers` | 2 | layers of each encoder |
| `num_mixing_enc_layers` | 2 | layers of the mixer, the stack that writes the global memory |
| `num_orth_dec_layers`, `num_phon_dec_layers` | 2 | layers of each decoder |

A head needs two dimensions because position is a rotation of pairs of them. A head of
one has no pair, so the encoders would read "cat" and "act" as one word. The smallest
size, `d_model=2` with `nhead=1`, validates and still cannot read order.

The scientific premise is that phonemes sharing phonetic features share embedding mass. A
phoneme's embedding is the mean of its active feature embeddings, never a free parameter of
its own. See `docs/decisions/0002-phoneme-row-ids-replace-ragged-feature-lists.md`.

## An example is an input and a target

A training example is two strings. The model reads the **input** and is taught to write
the **target**. They are different things by design. They may be equal: repetition is a
dataset whose two columns match, and no code treats that case differently.

Think of a flashcard. The front is what you read, and the back is what you must say.

Five types in `bridge/domain/datamodels/encodings.py` hold the tensors of an example. Each
is named for its side, and each tensor has exactly one reader.

| type | holds | read by |
|---|---|---|
| `Tokens` | `ids`, `pad_mask` | an encoder |
| `Target` | `ids`, `pad_mask`, `labels` | a decoder is fed `ids` and `pad_mask`; the loss and the metrics score against `labels` |
| `BridgeInput` | `orth` and `phon`, each a `Tokens` or `None` | the encoders |
| `BridgeTarget` | `orth` and `phon`, each a `Target` | the decoders, the loss and the metrics |
| `BridgeBatch` | `input`, a `BridgeInput`, and `target`, a `BridgeTarget` | `TrainingPipeline.forward` |

`orth` and `phon` are the names `PATHWAY_IO` gives the two modalities.

```
input.orth.ids        (batch, seq) int64      the orthography encoder
input.orth.pad_mask   (batch, seq) bool       the orthography encoder, True where padding
input.phon.ids        (batch, seq) int64      the phonology encoder
input.phon.pad_mask   (batch, seq) bool       the phonology encoder
target.orth.ids       (batch, seq) int64      the orthography decoder
target.orth.pad_mask  (batch, seq) bool       the orthography decoder
target.orth.labels    (batch, seq) int64      the loss and the metrics
target.phon.ids       (batch, seq) int64      the phonology decoder
target.phon.pad_mask  (batch, seq) bool       the phonology decoder
target.phon.labels    (batch, seq, 35) int64  the loss and the metrics
```

A pathway reads the input tensors of each modality it reads and the target tensors of each
modality it writes, and no others. `o2p` reads `input.orth` and `target.phon`.
`tests/application/training/test_input_target_wiring.py` pins this for every tensor and
every pathway: it rolls one tensor along the batch, and the loss must move exactly when
the pathway reads that tensor.

**Both modalities have the same shape.** Orthography and phonology are both
`(batch, sequence)` integer ids with a matching boolean pad mask. The labels are the one
asymmetry, and it is real. An orthographic label is a character id. A phonological label
is a vector of 35 classes, one for each feature column but `[PAD]`.

**Labels sit one position on from the ids.** `labels[:, i]` is what the decoder must emit
after reading `ids[:, :i + 1]`.

```
input.orth.ids       [EN] [BOS] c a t [EOS]
target.orth.ids      [EN] [BOS] c a t
target.orth.labels   [BOS] c a t [EOS]
```

So `[BOS]` predicts the first character, and the last character predicts `[EOS]`.
Phonology has the same layout without the language token. The tokenizer builds the labels
beside the ids, and the loss and the metrics read them as they are, with no shift of their
own. See `docs/decisions/0004` and `docs/decisions/0016`.

**Each type checks itself when it is built.**

- `Tokens`: `ids` and `pad_mask` are two-dimensional, with one shape and one device.
- `Target`: the same, and `labels` are int64 and hold one row per decoder position.
- `BridgeInput`: it carries at least one modality, each a `Tokens` passed by name,
  agreeing on batch size and device.
- `BridgeTarget`: it carries both modalities, each a `Target`, agreeing on batch size and
  device.
- `BridgeBatch`: `input` is a `BridgeInput` and `target` is a `BridgeTarget`, agreeing on
  batch size and device.

An input passed where a target belongs therefore fails where it is assembled.
`BridgeBatch(input, input)` raises `TypeError`, and so does `Model.generate` handed a
`BridgeTarget`.

**A modality the input does not carry is `None`.** A nonword has no pronunciation, so
`encode_input` returns it with `phon=None`. Phonemes given without a spelling are
`BridgeInput(phon=...)`. A pathway asked to read a missing modality raises `ValueError`
naming the pathway and the modality, in training and in generation:
`pathway 'p2o' reads phon, which this input does not carry`. A target always carries both
modalities, so `encode_target` returns `None` when a word has no pronunciation.

**`BridgeTokenizer` encodes each side with its own call.**

| call | returns |
|---|---|
| `encode_input(texts, languages)` | a `BridgeInput` |
| `encode_target(texts, languages)` | a `BridgeTarget`, or `None` |
| `orth_start(languages)` | the `(batch, 2)` `[LANG, BOS]` rows that start orthographic generation |

`texts` is a list of strings, one per row, and a bare string raises `TypeError`.

## Every word has a language

Each word of a string carries a language code: `EN`, `ES`, or `--` for a row whose
language was never given. `languages` is a list or tuple with one entry per row. An entry
is one code for the whole row (`"ES"`), one code per word (`"EN EN ES"`, or a list or
tuple of codes), or `None` for `--`. A tokenizer call with no `languages` gives `--` on
every row. One string for all the rows raises `ValueError`.

`bridge/domain/tokenizer/languages.py` owns the rules, and both tokenizers and the dataset
read a row through it. A word is a run of non-whitespace. One code covers every word;
otherwise the code count must equal the word count. An unsupported code, a wrong count, or
`--` beside a real code raises `ValueError`. An unsupported code gives the same message
from every entry point, and so does a wrong count.

The two modalities use the language differently.

- **Phonology** looks each word up in its own language's lexicon, by its position in the
  row. A word missing from that lexicon has no pronunciation, whichever other lexicon
  holds it. `--` reads the English lexicon. Phonology carries no language token.
- **Orthography** opens with the first word's language token and writes a language token
  again only where the language changes, directly before that word's first character.

```
the word is gato        EN EN EN ES
[EN] [BOS] t h e _ w o r d _ i s _ [ES] g a t o [EOS]
```

The language tokens are ordinary entries in the orthographic vocabulary. A mid-row switch
sits in a target's ids and in its labels, so it is a label the decoder is trained to emit,
and the model needs nothing new to carry it. `decode_with_languages` reads a row back as
its text and each word's language.

Orthographic accuracy is reported as two pairs. `letter_wise_accuracy` and
`word_wise_accuracy` count every scored token, switches included, which are the positions
the loss trains. The `_without_switches` pair leaves out the positions whose label is a
language token, so it measures spelling alone. `VocabSpec.orth_language_ids` names the
language tokens and is required, so both pairs are always reported. See
`docs/decisions/0015`.

## A dataset row is an input and a target

`BridgeDataset` (`bridge/domain/data/bridge_dataset.py`) reads a csv or a pkl file with
four columns.

| column | holds | when the column is missing |
|---|---|---|
| `word_raw` | the input string | the load raises `KeyError` |
| `language` | the input's languages: one code, or one code per word | every word is English |
| `target_raw` | the target string | the load raises `KeyError`, and the message names the fix |
| `target_language` | the target's languages, in the same form | a row whose `language` is one code gives its target that code |

`target_raw` is required so that a misspelt column cannot train repetition in silence. For
repetition, copy `word_raw` into `target_raw`. A per-word `language` says nothing about
the target's words, so a row with one and no `target_language` raises `ValueError` at
load, naming the row. A misspelt `language` or `target_language` header is not noticed:
the column counts as missing, and the last column of the table applies.

Every language cell, on both sides, is checked when the file loads. A blank cell, `--`, an
unsupported code or a wrong count raises `ValueError`, naming the row and the column. A
blank `word_raw` or `target_raw` cell raises the same way, and that is what a row missing
its last field holds. A local csv whose rows hold one more field than its header also
raises at load. That is what a comma between two codes does, and pandas would otherwise
read each row's first field as an index and load the codes as the words. A csv read from a
`gs://` path, through the storage client the caller passes, is not checked this way: there
the codes load as the words.

`dataset.examples` is one list holding one `Example` per row: `input`, `input_languages`,
`target` and `target_languages`. `shuffle` moves whole examples, so an input keeps its
target and both keep their languages.

`dataset[i]` takes an `int` or a `slice` and returns a `BridgeBatch`. Rows are read by
position only, since one spelling can sit under two languages and one input beside two
targets. Nothing is encoded at load and nothing is kept between reads. Pronunciations are
therefore not checked at load: a word with none, in the input or in the target, raises
`RuntimeError` when its row is read, wherever the row sits, naming the row and the side.
The dataset refuses such a row whichever pathway is training. A slice that selects no
rows raises `IndexError`. A nonword input reaches a model through
`BridgeTokenizer.encode_input`, not through a dataset.

The two kinds of message number rows differently. A load names a row of the file. A read
names a position in `dataset.examples`. They differ after a shuffle, and after a row whose
`word_raw` or `target_raw` is not a string, which the load leaves out with a warning.

## Two integer id spaces

Both are `torch.long`. Only naming separates them, and confusing them yields silently wrong
output rather than an error.

- **Row space**, 0 to 90, indexes rows of the phoneme table: *which phoneme*. This is what
  the phonological `ids` hold, in `BridgeInput.phon` and in `BridgeTarget.phon`, and what
  `Model.embed_phon_tokens` consumes.
- **Feature space**, 0 to 35, indexes phonetic feature columns: *which feature*. This is what
  `GenerationOutput.phon_tokens`, `PhonemeTokenizer.decode`, `VocabSpec.phon_pad_id` and the
  phonological `labels` use.

Convert with `PhonemeTable.features_of(row)` and `PhonemeTable.row_of(phoneme)`.

## The phoneme table

`bridge/core/phonreps.py` owns the feature scheme and is the single source of truth. It builds
a `PhonemeTable` from `bridge/core/phonreps.csv`: a `(91, 36)` multi-hot matrix of 91 phonemes
by 31 phonetic features plus 5 special-token columns (`[BOS]`, `[EOS]`, `[UNK]`, `[SPC]`,
`[PAD]`). Loading is cached per device for the process lifetime.

`PhonemeTable.fingerprint` is a digest of the table, recorded into `VocabSpec` and checked when
a checkpoint loads, so a `phonreps.csv` edit that silently relabels every id is detectable.

## Position and length

A token embeds to the same vector wherever it sits. Position enters the model in one
place, `rotate` in `bridge/domain/model/layers.py`. Inside self-attention it turns every
query and every key by an angle proportional to its position, before the dot product.
This is rotary position embedding. The score between two tokens then depends on how far
apart they are, and not on which slots they fill.

Think of the two hands of a clock. Move both on by an hour and the angle between them is
what it was.

| attention | position |
|---|---|
| self-attention of the two encoders | rotated |
| self-attention of the two decoders | rotated |
| a decoder's cross-attention to the global memory | none: the memory's rows have no order |
| the mixer | none: it is torch's stock `nn.TransformerEncoder` |
| the two cross-attentions between modalities on `op2op` | none: torch's stock `nn.MultiheadAttention` |

The encoders and the decoders are built from the blocks in `layers.py`. Torch's stock
layers give no access between the query and key projections and the dot product, which
is where the rotation sits. The blocks keep the parameter names and shapes of torch's
stacks. With the rotation switched off they compute what the stock stacks compute on the
same weights, and `tests/domain/model/test_rotary_layers.py` pins that.

**The model holds no maximum length.** No table is indexed by position. A sequence of any
length trains, evaluates and generates, and no length is checked anywhere.
`tests/domain/model/test_any_length.py` runs rows past 60 tokens on every pathway.

Three things still bound length, and none of them is a check.

- **What the model trained on.** A model reads an input somewhat longer than any it
  trained on. Trained on phrases of up to 19 tokens, a model that hears a phrase and
  spells its last word scores 0.848 [0.450, 1.000] at 23 to 30 tokens, over six seeds.
  It does not repeat a longer phrase: exact match is 0.000 to 0.003 past the trained
  lengths. Training data must cover the lengths in use.
- **The global memory.** A decoder reads `d_embedding` rows, however long the input is.
  With the default of 2, a model trained on phrases of up to 47 tokens repeats none of
  those of 33 to 47. With 8 rows it repeats 0.620 of them, and with 32 rows 0.895. A run
  on phrases sets `d_embedding`.
- **The cap at generation.** Nothing in the model says when to stop, so `generate` takes
  `max_new_tokens` from its caller. "How a batch flows" says how it counts.

Measured 2026-11-24 on CPU, on small closed vocabularies. `docs/decisions/0017` carries
the numbers and names the scripts.

## Layout

```
bridge/core/phonreps.py               the feature scheme and PhonemeTable
bridge/core/pronunciation_lexicons/   per-language word to phoneme dictionaries
bridge/domain/datamodels/             VocabSpec, ModelConfig, GenerationOutput, TrainingEvent
bridge/domain/datamodels/encodings.py Tokens, Target, BridgeInput, BridgeTarget, BridgeBatch
bridge/domain/datamodels/pathways.py  PATHWAY_IO: what each pathway reads and writes
bridge/domain/tokenizer/              CharacterTokenizer, PhonemeTokenizer, BridgeTokenizer
bridge/domain/tokenizer/languages.py  what a word is, and which language each word is in
bridge/domain/model/layers.py         rotate, and the encoder and decoder blocks whose
                                      self-attention reads position through it
bridge/domain/model/model.py          token embeddings, the mixer, the five pathways,
                                      generation loops
bridge/domain/data/bridge_dataset.py  dataset of input and target rows, languages checked at load
bridge/application/training/          TrainingPipeline, loss, metrics, checkpoints
bridge/utils/device_manager.py        the process device, selected by BRIDGE_DEVICE
tests/fixtures/phon_baseline.pt       golden master, immutable
tests/fixtures/bench_phon.py          phonological hot-path benchmarks
tests/application/training/test_input_target_wiring.py
                                      asserts which tensors of a batch each pathway reads
tests/domain/model/test_rotary_layers.py
                                      the blocks against torch's stock stacks, and the rotation
tests/domain/model/test_any_length.py position, padding and generation on the whole model
tests/domain/model/test_max_new_tokens.py
                                      the cap on what generation writes
tests/test_documentation.py           asserts this document against the live system
tests/test_diagram.py                 asserts BRIDGE-Models.drawio against the live model
tests/fixtures/sync_diagram_table.py  rewrites the figure's feature table from phonreps.csv
docs/decisions/                       architecture decision records
```

`BRIDGE-Models.drawio`, at the repo root, draws the data flow of each pathway, one page per
pathway, and is served on the model's website. Each page fades the blocks its pathway does
not run. `tests/test_diagram.py` hooks every submodule, records which ones each pathway's
training forward fires, and fails when a page draws a block opaque that its pathway does not
run, or the reverse. The figure draws the input and the target as separate boxes of words.
The test also fails when an encoder is not drawn reading an input box, or a decoder is not
drawn fed from a target box.

The library performs no I/O of its own: no metrics sink, no cloud client, no logger. It
reads the dataset path it is given and writes the checkpoint path it is given, and
everything else reaches the caller through the `TrainingEvent` stream. See
`docs/decisions/0010`.

## How a batch flows

Training, `o2p` as the example:

```
a file row: input string, target string, one language code per word of each
  -> BridgeDataset[slice]             -> BridgeBatch
       BridgeTokenizer.encode_input     -> batch.input, a BridgeInput
       BridgeTokenizer.encode_target    -> batch.target, a BridgeTarget
  -> TrainingPipeline.forward(batch)  -> Model.forward(task="o2p", ...)
       encoder tensors                  <- batch.input.orth
       decoder tensors                  <- batch.target.phon
  -> Model.embed_o                    -> global memory (batch, d_embedding, d_model)
       orthography_encoder              token embeddings in; self-attention rotated by position
       transformer_mixer                the global rows and the encoding in; no position
  -> Model._decode_phon               -> logits["phon"] (batch, 2, seq, 35)
       phonology_decoder                causal self-attention rotated by position;
                                        cross-attention to the global memory, no position
  -> TrainingPipeline.compute_loss(logits, batch.target)
                                      -> cross_entropy against batch.target.phon.labels,
                                         ignore_index=phon_pad_id
```

`TrainingPipeline.forward` picks the tensors by a query against `PATHWAY_IO`: the ids and
mask of each modality the pathway reads, from `batch.input`, and of each modality it
writes, from `batch.target`. `Model.forward` takes those tensors, not a batch. The decoder
reads the global memory, whose size does not depend on the input's length, so an input and
its target need not be the same length.

`compute_loss(logits, target)` and `compute_metrics(logits, target)` score the logits
against `target.orth.labels` and `target.phon.labels`. `single_step` and `evaluate` take a
dataset and a slice, and do all of the above.

Embedding the phonological side is a table lookup, not a loop: `phon_feature_matrix @
phonology_embedding.weight` gives every phoneme's embedding in one small matmul, and
`F.embedding` indexes it. The matrix is a non-persistent buffer, so it stays out of
`state_dict`.

Generation is
`Model.generate(inputs, pathway, *, max_new_tokens, orth_start=None, deterministic=False)`.
It reads a `BridgeInput` and nothing else: there is no target, and nothing about the
output is taken from the input in silence. It replaces the decoder call with
`orthography_decoder_loop` or `phonology_decoder_loop`. Each writes one position per step
and decodes the whole prefix to do it, so a token keeps the position it will have in the
finished sequence. They return ragged per-item lists because sequences finish at
different lengths.

`max_new_tokens` is the most tokens to write after the start of each sequence. It is
keyword-only and required, with no default: the model holds no length to fall back on.
The start is `[LANG, BOS]` for orthography and `[BOS]` for phonology, and it is not
counted.

| point | rule |
|---|---|
| The value | A positive `int`. `0`, `2.0`, `True` and a numpy integer each raise `ValueError` naming the value and its type |
| Stopping | A sequence stops at `[EOS]` or at the cap, whichever comes first |
| A finished row | It holds `[EOS]`, and padding after it (`docs/decisions/0009`) |
| A cut row | It reached the cap and holds no `[EOS]`. That is how a caller sees it was cut |
| `op2op` | The one cap applies to each modality written |
| Sizing, orthography | `n` characters, spaces included, take `n + 1` tokens, and one more for each change of language |
| Sizing, phonology | One token for each phoneme, one for each gap between words, and one for `[EOS]` |
| Cost | The batch is dense, so one row that never ends runs every row to the cap |

A phonological row counts as finished when the `[EOS]` feature is on at one of its
positions, alone or beside other features. A cut row can decode to the same string as a
finished shorter one, so test the ids, not the text. See `docs/decisions/0017`.

A pathway that writes orthography requires `orth_start`, the `(batch, 2)` `[LANG, BOS]`
rows built by `BridgeTokenizer.orth_start(["ES", ...])`, one code per row. Omitting it
raises `ValueError`. It names the language of the first word written, and any later
change of language is the decoder's to emit. These are the two positions training feeds
the orthographic decoder first. `generate` checks the shape, dtype, device and id range of
`orth_start`, not which ids it holds. A pathway that writes no orthography does not read
it. See `docs/decisions/0008` and `docs/decisions/0016`.

## Who drives training

The library owns the step; the caller owns the loop, and now in fact rather than only in
principle. `TrainingPipeline.single_step` runs one optimizer step over one slice and
`train_steps(dataset, slices, epoch)` yields a `TrainingEvent` after each of them. It takes
the partition rather than building one: which slices a run uses, how many epochs it does,
whether to shuffle between them, and what to record are the caller's. See
`docs/decisions/0013`.

`TrainingEvent` and `TrainingPhase` remain the shared vocabulary for a run. The library
emits `train` events; a caller's loop emits `validation`, `test` and `epoch` ones. The
metrics of a `train` event carry the losses, and under `input` and `target` the strings of
the rows the step covered, each a JSON list.

`save_checkpoint(path, epoch)` takes a destination rather than a run name and a cadence, so
where a run's weights land is the caller's decision. See
`docs/decisions/0006-the-caller-owns-the-training-loop.md`.

**A checkpoint loads only into a model with the same tensors.** `load_model(path)`, and a
pipeline built with `TrainingConfig.checkpoint_path`, compare the file with the model
before copying anything: every tensor name, and every shape. On any difference they
raise `RuntimeError` naming the file, how many tensors differ and the first of each
kind, and the model, the optimizer and `start_epoch` are left as they were. Nothing loads
with keys missing.

A checkpoint written by a model with position tables is refused this way, which covers
one saved at commit `a3cb285`, the last commit with the tables. There is no migration:
those weights were trained to read position from the tables, and would mean nothing
here. A checkpoint whose `d_embedding` or `d_model` differs from the model's is refused
by its shapes. See `docs/decisions/0017`.

## Dependencies

PyTorch for the model, pydantic v2 for configs and validation, pandas for the feature CSV,
and numpy. `uv` for environment and task running, pytest, mypy and ruff for checks.
Nothing else is installed: the library performs no I/O of its own, so it ships no cloud or
experiment-tracking client. See `docs/decisions/0010`.

## Known defects

Tracked as GitHub issues rather than restated here:

- **#233** `Model.generate` fails opaquely on one remaining input shape, the second of the
  three the issue lists. A half-precision model cannot generate on the three pathways
  that write orthography. `GenerationOutput` rejects it with
  `orth_probs probabilities must sum to 1`, because the check's `atol=1e-5` is a float32
  tolerance applied to a float16 sum. It generates on `o2p` and `p2p`. The other two
  shapes are refused at the boundary by name. A zero-row batch raises
  `orth_enc_input has no rows; there is nothing to generate`, or the same for
  `phon_enc_input` (`docs/decisions/0011`). An input that lacks a modality its pathway
  reads, such as phonemes alone on `o2p`, raises
  `pathway 'o2p' reads orth, which this input does not carry` (`docs/decisions/0016`).
  Measured 2026-11-24 on CPU, on the model with rotary position; record 0017 names the
  script

## Decision index

| record | decision |
|---|---|
| [0001](decisions/0001-record-architecture-decisions.md) | Record architecture decisions as numbered immutable files |
| [0002](decisions/0002-phoneme-row-ids-replace-ragged-feature-lists.md) | Phonology uses `(batch, seq)` row ids, matching orthography, instead of ragged feature lists |
| [0003](decisions/0003-equivalence-by-differential-against-a-noise-floor.md) | Behavioural equivalence is proven by differential against a pristine baseline, judged against a measured noise floor |
| [0004](decisions/0004-orthographic-teacher-forcing-alignment.md) | The orthographic loss and metrics score each decoder position against the next token, so `[BOS]` predicts the first character |
| [0005](decisions/0005-phonological-metrics-take-the-pad-id.md) | Phonological metrics take `phon_pad_id` as a required argument; every metric reported before this is void |
| [0006](decisions/0006-the-caller-owns-the-training-loop.md) | The caller owns the training loop and decides when to checkpoint; the library owns the step |
| [0007](decisions/0007-model-device-is-derived-not-stored.md) | `Model.device` is derived from a parameter, so `.to()` is authoritative and the model cannot misreport where it is |
| [0008](decisions/0008-generation-seeds-the-training-prefix.md) | Orthographic generation is seeded with `[LANG, BOS]`, the prefix training uses, keeping the language token usable |
| [0009](decisions/0009-a-finished-sequence-emits-padding.md) | A finished sequence emits padding, so generation output ends cleanly; only the accepted sequence is masked |
| [0010](decisions/0010-the-library-ships-no-io.md) | The library computes and the caller does the I/O; `bridge/infra/` is gone and no cloud SDK is installed |
| [0011](decisions/0011-one-table-says-what-each-pathway-reads-and-writes.md) | `PATHWAY_IO` is the single definition of what each pathway reads and writes; validation is uniform across pathways |
| [0012](decisions/0012-the-pipeline-follows-the-model.md) | `TrainingPipeline.device` reads `model.device`; the pipeline never moves the model |
| [0013](decisions/0013-the-caller-owns-the-loop-in-fact.md) | The library keeps the step and loses the loop; split, batching, shuffling, epochs and progress bars are the caller's |
| [0014](decisions/0014-a-config-that-validates-runs-every-pathway.md) | Every size in `ModelConfig` is at least one and `nhead` divides `d_model`, so a config that validates builds a model that runs every pathway. 0017 adds that each head has two dimensions or more |
| [0015](decisions/0015-language-belongs-to-each-word.md) | Language belongs to each word of a row, by position; phonology reads each word in its own lexicon and orthography marks each change |
| [0016](decisions/0016-a-training-example-is-an-input-and-a-target.md) | A training example is an input and a target, held in five types named for their side; `target_raw` is required, and the caller names the language orthographic generation starts in |
| [0017](decisions/0017-positions-are-computed-for-any-length.md) | Position is a rotation inside the self-attention of the encoders and decoders, computed for any length; generation takes a required `max_new_tokens`; `d_embedding` defaults to 2; checkpoints from a model with position tables are refused |
