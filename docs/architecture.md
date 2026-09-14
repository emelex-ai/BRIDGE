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

The scientific premise is that phonemes sharing phonetic features share embedding mass. A
phoneme's embedding is the mean of its active feature embeddings, never a free parameter of
its own. See `docs/decisions/0002-phoneme-row-ids-replace-ragged-feature-lists.md`.

## Both modalities have the same shape

This is the load-bearing structural fact. An `EncodingComponent`
(`bridge/domain/datamodels/encodings.py`) carries `(batch, sequence)` integer ids plus a
matching boolean pad mask, for orthography and phonology alike. A `BridgeEncoding` pairs one
of each and validates that they agree on batch size and device.

```
enc_input_ids   (batch, seq) int64     ids fed to the encoder
enc_pad_mask    (batch, seq) bool      True where padding
dec_input_ids   (batch, seq) int64     ids fed to the decoder
dec_pad_mask    (batch, seq) bool
targets         (batch, seq, 35)       phonological only, loss targets
```

`targets` is the only asymmetry, and it is real: phonology carries explicit loss targets while
orthography derives them from `enc_input_ids`.

## Two integer id spaces

Both are `torch.long`. Only naming separates them, and confusing them yields silently wrong
output rather than an error.

- **Row space**, 0 to 90, indexes rows of the phoneme table: *which phoneme*. This is what
  `PhonemeTokenizer.encode` emits and what `Model.embed_phon_tokens` consumes.
- **Feature space**, 0 to 35, indexes phonetic feature columns: *which feature*. This is what
  `GenerationOutput.phon_tokens`, `PhonemeTokenizer.decode`, `VocabSpec.phon_pad_id` and the
  loss targets use.

Convert with `PhonemeTable.features_of(row)` and `PhonemeTable.row_of(phoneme)`.

## The phoneme table

`bridge/core/phonreps.py` owns the feature scheme and is the single source of truth. It builds
a `PhonemeTable` from `bridge/core/phonreps.csv`: a `(91, 36)` multi-hot matrix of 91 phonemes
by 31 phonetic features plus 5 special-token columns (`[BOS]`, `[EOS]`, `[UNK]`, `[SPC]`,
`[PAD]`). Loading is cached per device for the process lifetime.

`PhonemeTable.fingerprint` is a digest of the table, recorded into `VocabSpec` and checked when
a checkpoint loads, so a `phonreps.csv` edit that silently relabels every id is detectable.

## Layout

```
bridge/core/phonreps.py               the feature scheme and PhonemeTable
bridge/core/pronunciation_lexicons/   per-language word to phoneme dictionaries
bridge/domain/datamodels/             EncodingComponent, BridgeEncoding, VocabSpec,
                                      ModelConfig, GenerationOutput
bridge/domain/tokenizer/              CharacterTokenizer, PhonemeTokenizer, BridgeTokenizer
bridge/domain/model/model.py          encoders, decoders, generation loops
bridge/domain/data/bridge_dataset.py  dataset, language resolution, encoding memo
bridge/application/training/          TrainingPipeline, loss, metrics
bridge/infra/                         metrics loggers, GCS and wandb clients
tests/fixtures/phon_baseline.pt       golden master, immutable
tests/test_documentation.py           asserts this document against the live system
docs/decisions/                       architecture decision records
```

## How a batch flows

Training, `o2p` as the example:

```
word strings
  -> BridgeTokenizer.encode           -> BridgeEncoding (both modalities)
  -> BridgeDataset[slice]
  -> TrainingPipeline.forward         -> Model.forward(task="o2p", ...)
  -> Model.embed_o                    -> global memory (batch, d_embedding, d_model)
  -> Model._decode_phon               -> logits["phon"] (batch, 2, seq, 35)
  -> TrainingPipeline.compute_loss    -> CrossEntropyLoss(ignore_index=phon_pad_id)
```

Embedding the phonological side is a table lookup, not a loop: `phon_feature_matrix @
phonology_embedding.weight` gives every phoneme's embedding in one small matmul, and
`F.embedding` indexes it. The matrix is a non-persistent buffer, so it stays out of
`state_dict` and old checkpoints still load with `strict=True`.

Generation replaces the decoder call with `orthography_decoder_loop` or
`phonology_decoder_loop`, which decode one position at a time and return ragged per-item
lists because sequences finish at different lengths.

## Who drives training

The library owns the step; the caller owns the loop. `TrainingPipeline.single_step` runs one
optimizer step over one slice, `train_steps(epoch)` yields after each of them, and
`run_train_val_loop` is a thin wrapper that adds shuffling, validation and an epoch summary.
It emits a `TrainingEvent` per step and per boundary, tagged `train`, `validation`, `test` or
`epoch`, and writes no checkpoints of its own.

`save_checkpoint(path, epoch)` takes a destination rather than a run name and a cadence, so
where a run's weights land is the caller's decision. See
`docs/decisions/0006-the-caller-owns-the-training-loop.md`.

## Dependencies

PyTorch for the model, pydantic v2 for configs and validation, pandas for the feature CSV,
`uv` for environment and task running, pytest, mypy and ruff for checks. Optional Google Cloud
Storage and Weights and Biases clients under `bridge/infra/`.

## Known defects

Tracked as GitHub issues rather than restated here:

- **#228** generation seeds `[BOS]` at decoder position 0 while training places it at
  position 1, after `[LANG]`, so it carries a different position embedding in each regime

## Decision index

| record | decision |
|---|---|
| [0001](decisions/0001-record-architecture-decisions.md) | Record architecture decisions as numbered immutable files |
| [0002](decisions/0002-phoneme-row-ids-replace-ragged-feature-lists.md) | Phonology uses `(batch, seq)` row ids, matching orthography, instead of ragged feature lists |
| [0003](decisions/0003-equivalence-by-differential-against-a-noise-floor.md) | Behavioural equivalence is proven by differential against a pristine baseline, judged against a measured noise floor |
| [0004](decisions/0004-orthographic-teacher-forcing-alignment.md) | The orthographic loss and metrics target `enc_input_ids[:, 1:]`, so `[BOS]` predicts the first character |
| [0005](decisions/0005-phonological-metrics-take-the-pad-id.md) | Phonological metrics take `phon_pad_id` as a required argument; every metric reported before this is void |
| [0006](decisions/0006-the-caller-owns-the-training-loop.md) | The caller owns the training loop and decides when to checkpoint; the library owns the step |
| [0007](decisions/0007-model-device-is-derived-not-stored.md) | `Model.device` is derived from a parameter, so `.to()` is authoritative and the model cannot misreport where it is |
