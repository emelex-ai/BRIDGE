# Spec: Phonological fast path, one phoneme–feature table with four uses

**Issue:** [#221](https://github.com/emelex-ai/BRIDGE/issues/221) (plus its six "related findings")
**Status:** Implemented, supersedes the first draft of this spec

Three things shipped differently from the sketch below. `PhonemeTable.normalized` (§ below)
became the module function `row_normalize`. `Model._phon_positions(seq_len, offset)` became the
modality-agnostic `_positions(table, seq_len, offset)`, shared with `embed_orth_tokens`, and
gained a bounds check. "training_pipeline.py needs zero changes" turned out false: it swapped
`load_phonreps_array` for `load_phoneme_table(...).phonetic_features` and gained the checkpoint
fingerprint warning.
**Baseline commit:** `5d7e8d9`
**Measured on:** RTX 5080 / 12-core CPU, `torch.set_num_threads(1)`, torch 2.12.0+cu130

---

## 0. Audit verdict

The first draft of this spec adopted issue #221's proposal verbatim: emit padded
`(B, L, F)` feature ids + mask from the tokenizer, and make `embed_phon_tokens` a masked
gather. **Measurement says that is the worst of the four viable vectorized approaches, on
both axes.**

Three findings reverse the recommendation:

1. **The padded-gather approach materializes a `(B, L, F, d_model)` intermediate.** At
   `B=1024, L=25, d=128` that is 92 MB per call, and it is 85× slower forward / 29× slower
   fwd+bwd than the best alternative on CPU. It is the only candidate whose cost grows with
   `F`, and it degrades with batch size and `d_model`, the two knobs this project will turn.
2. **The averaging is linear in the embedding weight, and the set of phonemes is finite and
   known at init.** So `mean(W[features(p)])` for all `P = 91` phonemes is one small constant
   matrix times `W`. Precomputing that table per forward pass and gathering `(B, L)` phoneme
   ids is both the fastest option and the one that makes the phonological encoding
   *structurally identical to the orthographic one*, a plain `(B, L)` LongTensor.
3. **The same table also fixes bottlenecks the issue does not list.** `PhonemeTokenizer.encode`
   builds its `targets` tensor with a per-position `torch.isin` device write; replacing it
   with a gather from the same table is **10–15× faster on CPU and 31–61× on CUDA**.
   Tokenization is 12.1% of a CUDA train step, and nobody had flagged it.

The recommended approach (**C**, below) is therefore *faster* than issue #221's proposal
**and** deletes the ragged `list[list[Tensor]]` representation rather than adding a second
representation beside it. `bridge/application/training/training_pipeline.py` needs **zero
changes**, and `Model.embed_phon_tokens` keeps its current one-argument signature.

No parameter shape changes, so **existing checkpoints load unchanged** (§6.4).

---

## 1. Measured baseline

End-to-end o2p train step, real data (`tests/domain/model/data/data.csv`), `B=32`, `L=7`,
`d_model=64`:

| | CPU | CUDA |
|---|---|---|
| full train step | 35.8 ms | 46.8 ms |
| - tokenize (`dataset[slice]`) | 1.13 ms (3.1%) | 5.65 ms (**12.1%**) |
| - forward total | 16.45 ms (45.9%) | 15.34 ms (32.8%) |
| - of which `embed_phon_tokens` | 6.22 ms (**17.4%**) | 10.64 ms (**22.7%**) |
| - backward + optimizer | 18.26 ms (50.9%) | 25.77 ms (55.1%) |
| `generate()` o2p, deterministic | 364 ms | 16.4 ms |

The `embed_phon_tokens` shares reproduce issue #221's 17%/25%. The forward-only share
understates the cost: the `B × L` autograd nodes are paid again in backward.

`BridgeTokenizer()` construction, measured: **993 ms, 81 MB retained**, and the periodic
`gc.collect()` in `TrainingPipeline` goes from **109 ms → 188 ms** when a second one exists.
This confirms related finding 6 exactly.

---

## 2. Candidates

All take a representation **prebuilt by the tokenizer**: issue #221 established that building
padding per call recovers only ~2× and just relocates the bottleneck.

| | Approach | Tokenizer emits |
|---|---|---|
| **0** | current: Python double loop | ragged `list[list[Tensor]]` |
| **A** | padded gather + masked mean *(issue #221's proposal)* | `(B,L,F)` ids + `(B,L,F)` bool mask |
| **B** | normalized dense multi-hot GEMM: `mh_norm @ W` | `(B,L,V)` float multi-hot |
| **C** | **phoneme table: `F.embedding(rows, A_norm @ W)`** | **`(B,L)` long phoneme row ids** |
| **E** | `F.embedding_bag(flat, W, offsets, mode='mean')` | flat ids + offsets (2× 1-D) |
| **D** | *reparameterize:* make phonemes the embedding table | - **rejected, see §2.3** |

### 2.1 Real-data shape (`B=32, L=7, F=7, V=36, d=64`)

Forward only / forward+backward, ms, and speedup vs current:

| | CPU fwd | CPU fwd+bwd | CUDA fwd | CUDA fwd+bwd |
|---|---|---|---|---|
| 0 current | 5.899 - | 12.580 - | 10.776 - | 29.564 - |
| A padded | 0.067 **88×** | 0.488 **26×** | 0.079 **137×** | 0.450 **66×** |
| B multihot | 0.023 **254×** | 0.322 **39×** | 0.046 **235×** | 0.425 **70×** |
| **C table** | **0.027 223×** | **0.321 39×** | **0.062 173×** | **0.439 67×** |
| E bag | 0.049 **121×** | 0.380 **33×** | 0.036 **299×** | 0.526 **56×** |

At this small shape **every** vectorized option clears issue #221's ">20× isolated forward"
bar with room to spare, and on CUDA all four are launch-bound and statistically
indistinguishable on fwd+bwd. **Speed does not discriminate here, invasiveness must.**

### 2.2 Scaling sweep (synthetic, `P=91`, `V=36`, `F≤7`): where they separate

CPU, ms, fwd / fwd+bwd:

| shape | A padded | B multihot | **C table** | E bag |
|---|---|---|---|---|
| B=32 L=7 d=64 | 0.062 / 0.264 | 0.019 / 0.117 | **0.020 / 0.125** | 0.042 / 0.183 |
| B=256 L=25 d=64 | 4.49 / 7.09 | 0.310 / 0.786 | **0.078 / 0.558** | 0.221 / 2.27 |
| B=1024 L=25 d=128 | 99.4 / 144.9 | 3.47 / 9.63 | **1.16 / 4.99** | 2.62 / 16.7 |
| B=256 L=25 d=512 | 91.2 / 144.9 | 3.10 / 9.73 | **1.25 / 8.01** | 3.78 / 10.3 |

CUDA, ms, fwd / fwd+bwd:

| shape | A padded | B multihot | **C table** | E bag |
|---|---|---|---|---|
| B=256 L=25 d=64 | 0.081 / 0.459 | 0.046 / 0.268 | 0.076 / 0.406 | 0.033 / 0.351 |
| B=1024 L=25 d=128 | 0.540 / 1.045 | 0.046 / 0.277 | 0.068 / 0.421 | 0.071 / 0.394 |
| B=256 L=25 d=512 | 0.533 / 0.946 | 0.038 / 0.240 | 0.069 / 0.407 | 0.077 / 0.321 |

Reproduced across two runs; per-cell noise is roughly ±20%, except one anomalous
`A / B=32 L=25 / fwd` cell (~1.1 ms) that reproduces but reports fwd > fwd+bwd, an allocator
artefact of the large intermediate, not a real forward cost. It does not affect any conclusion.

**A is the only candidate that blows up**, because it materializes `(B, L, F, d_model)` before
reducing. C and B never allocate anything wider than `(B, L, d_model)`.

### 2.3 Rejected: D, make phonemes the embedding table

Tempting (`nn.Embedding(91, d)` and the whole problem vanishes) and **wrong**. It deletes
the feature-sharing inductive bias, that phonemes with overlapping phonetic features share
embedding mass, which is the scientific premise of BRIDGE. Rejected on modelling grounds,
not performance grounds. C achieves the same runtime shape (a gather from a `(P, d)` table)
while keeping `W` as the parameter and the table as a *derived* quantity, so gradients still
flow into shared features.

---

## 3. Ranking

### By speedup (fwd+bwd, the number that matters for training)

1. **C table**: best at every CPU shape; within noise of B on CUDA. Alone in being
   *asymptotically* independent of `B·L` in its GEMM (`P=91` rows, not `B·L=25,600`).
2. **B multihot**: within 10–40% of C. Same primitive, applied per position instead of per
   phoneme.
3. **E bag**: 1 fused kernel, best CUDA forward, but 2–3× worse fwd+bwd at scale and its
   `flat + offsets` representation is hostile to the generation path.
4. **A padded** *(issue #221's proposal)*, 20–90× worse than C at scale on CPU, 2–10× worse
   on CUDA. Fine at today's toy shapes; the wrong thing to build into a hot path.

### By invasiveness (least → most)

1. **C table**: **net code deletion.** `(B, L)` long tensor is exactly the orthographic
   shape, so: `EncodingComponent.enc_input_ids: Any` becomes `torch.Tensor`; the ragged
   branch of `EncodingComponent.to()` disappears; `_validate_phonological_component` collapses
   into the orthographic validator; `BridgeEncoding.__post_init__`'s nested device walk
   disappears; `Model._validate_phon_inputs` collapses into `_validate_orth_inputs`.
   `embed_phon_tokens` keeps its **one-argument signature**, so no `forward_*` signature
   changes and **`training_pipeline.py` is untouched**.
2. **B multihot**: 1 field type change, but `(B, L, V)` float is a distinct shape from
   orthography, so the dual-representation validation code stays. No collapse.
3. **E bag**: 2 new fields (`flat`, `offsets`), and generation must convert dense
   `feature_presence` → ragged → flat every step, reintroducing the thing being removed.
4. **A padded**: 4 new fields on `EncodingComponent`, all 7 `forward_*`/`embed_*`/`_decode_*`
   signatures change from 1 phon argument to 2, `training_pipeline.forward()` changes at 6
   call sites, **and** the ragged representation stays alongside for a deprecation window, plus
   a `pad_phon_features` slow path and an `embed_phon_tokens_ragged` shim that need a review
   rule to stop them leaking back into hot paths. Strictly additive complexity.

**C wins both rankings.** The rest of this spec specifies C.

### The cost of C, stated honestly

~120 test assertion sites across 7 files reference the ragged phonological representation and
must change. That is real work, but A touches the same 7 files (it changes the same
signatures and adds fields) *and* leaves the codebase with two representations forever. C's
churn is substitution that ends with less code; A's is addition.

---

## 4. The unifying idea

A phoneme's embedding is the **mean of its active feature embeddings**: a *linear* function
of `W`:

```
embed(p) = (1/|features(p)|) · Σ_{f ∈ features(p)} W[f]  =  (A_norm @ W)[p]
```

where `A_norm` is a constant `(P, V)` row-normalized multi-hot matrix, `P = 91`
(86 phonreps rows + 5 special tokens), `V = 36`. `A_norm` is fixed configuration, it does not
train, and it is fully determined by `phonreps.csv`.

**One table, four uses:**

| Use | Operation | Replaces |
|---|---|---|
| Tokenizer: encoder/decoder inputs | `rows`, plain Python `int` lookups → one `torch.tensor(rows)` | per-phoneme tensor build + ragged list |
| Tokenizer: `targets` | `target_table[target_rows]` (one gather) | per-position `targets[i,j] = torch.isin(...)` device write |
| Model, training | `F.embedding(rows, A_norm @ W)` | the Python double loop |
| Model, generation | `(presence / counts) @ W` on the dense `feature_presence` **that `phono_sample` already computes** | re-embedding a growing ragged list every step |

The train and generate paths are the *same* linear op, training just precomputes it for the
finite phoneme set. Generation cannot, because a sampled feature vector need not be any real
phoneme; it uses candidate B directly on the dense presence matrix it already has.

---

## 5. Bottleneck inventory: every one resolved

| # | Bottleneck | Resolution | Phase |
|---|---|---|---|
| **#221** | `embed_phon_tokens` Python double loop, 17%/23% of a train step | §4 table gather | 3 |
| **1** | Generation re-embeds the whole prefix each step, `O(B·L²)` | Multi-hot step embedding + `position_offset` ⇒ `cat` one new `(B,1,d)` row instead of recomputing | 4 |
| **2** | `CharacterTokenizer.encode` element-by-element | *already fixed* on main | - |
| **3** | `arange` reallocated in nested loop | *already fixed* on main (`self._feature_range`) | - |
| **4** | Per-item GPU syncs in decode loops (`phono_sample` list build, `sequence_finished[b]`, `torch.tensor(new_tokens[b])`) | Everything stays dense `(B, …)`; ragged outputs built **once after** the loop | 4 |
| **5** | EOS check rescans full history every step, `O(B·L²)` device comparisons | Running `finished |= presence[:, eos_id]`, one `.all()` per step | 4 |
| **6** | Lexicon parsed twice: 993 ms + 81 MB, inflates `gc.collect()` 109→188 ms | `BridgeDataset(tokenizer=...)` injection; `TrainingPipeline` shares one | 1 |
| **7** | **(new, found in this audit)** `PhonemeTokenizer.encode` builds `targets` with a per-position `torch.isin` device write, tokenization is 12.1% of a CUDA step | Gather from the same table: **10–15× CPU / 31–61× CUDA** | 2 |

Finding 7 is free once the table exists, and was verified to reproduce `targets` **bitwise**
(§7.2).

---

## 6. Target design

### 6.1 `bridge/core/phonreps.py`: own the table

This module's docstring already declares it the tokenizer-independent home of the feature
scheme. Extend it so exactly one place defines row order:

```python
SPECIAL_TOKENS = ("[BOS]", "[EOS]", "[UNK]", "[SPC]", "[PAD]")

@dataclass(frozen=True)
class PhonemeTable:
    multihot: torch.Tensor        # (P, V) float 0/1, row order: phonreps.csv order, then SPECIAL_TOKENS
    row_index: dict[str, int]     # phoneme or special token -> row
    base_dim: int

    @property
    def normalized(self) -> torch.Tensor:   # A_norm; zero rows stay zero
        return self.multihot / self.multihot.sum(-1, keepdim=True).clamp(min=1)

    @property
    def fingerprint(self) -> str:           # sha256 of multihot bytes, see §6.4
        ...

def load_phoneme_table(device: torch.device | str = "cpu") -> PhonemeTable: ...
```

Row order is `phonreps.csv` order (0–85) then `SPECIAL_TOKENS` order (86–90). Defined once,
consumed by both tokenizer and model, the single contract that makes C safe.

### 6.2 `PhonemeTokenizer`: emit row ids

```python
# __init__
table = load_phoneme_table(device=self.device)
self._row_index = table.row_index
self._unk_row = table.row_index["[UNK]"]
self._target_table = table.multihot[:, : V - 1].long().contiguous()
# The [PAD] row differs between input and target tables: as a decoder *input* a padded slot
# embeds as one-hot([PAD]); as a *target* it must be the CrossEntropyLoss ignore_index so the
# position contributes no loss. Current code gets this by pre-filling `targets` with pad_id.
self._target_table[table.row_index["[PAD]"]] = self.special_token_dims["[PAD]"]
```

```python
# encode: the existing per-word loop, with int appends instead of tensor builds
for seq in word_phonemes:
    r = [self._row_index.get(p, self._unk_row) for p in seq]
    enc_rows.append([bos, *r, eos] + [pad] * (enc_length - len(r) - 2))
    dec_rows.append([bos, *r]      + [pad] * (dec_length - len(r) - 1))
    tgt_rows.append([*r, eos]      + [pad] * (dec_length - len(r) - 1))

enc_input_ids = torch.tensor(enc_rows, dtype=torch.long, device=self.device)
dec_input_ids = torch.tensor(dec_rows, dtype=torch.long, device=self.device)
targets       = self._target_table[torch.tensor(tgt_rows, dtype=torch.long, device=self.device)]
```

`_get_phoneme_indices` and its `vector_cache` are deleted, a dict lookup returning an `int`
replaces a dict lookup returning a `Tensor`. Keep `_get_phoneme_indices` only if
`phoneme_vector_to_phoneme` still needs it (it does not; it reads `phonreps_array` directly).

### 6.3 `Model`: table gather for training, GEMM for generation

```python
# __init__: persistent=False keeps it out of state_dict; see §6.4
self.register_buffer("phon_feature_matrix", load_phoneme_table(self.device).normalized,
                     persistent=False)

def _phon_positions(self, seq_len: int, offset: int) -> torch.Tensor:
    return self.phon_position_embedding.weight[offset : offset + seq_len][None]

def embed_phon_tokens(self, rows: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
    """(B, L) phoneme row ids -> (B, L, d_model). Training / teacher-forced path."""
    # A_norm @ W is (P=91, V=36) @ (V, d), recomputed per call because W trains.
    # ~210 KFLOP; measured cheaper than the per-position GEMM it replaces.
    table = self.phon_feature_matrix @ self.phonology_embedding.weight
    return F.embedding(rows, table) + self._phon_positions(rows.shape[1], position_offset)

def embed_phon_vectors(self, multihot: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
    """(B, L, V) dense feature vectors -> (B, L, d_model). Generation path, sampled
    vectors are not necessarily real phonemes, so no table row exists for them."""
    normalized = multihot / multihot.sum(-1, keepdim=True).clamp(min=1)
    averaged = normalized @ self.phonology_embedding.weight
    return averaged + self._phon_positions(multihot.shape[1], position_offset)
```

`embed_phon_tokens` keeps **one required argument**, so `_decode_phon`, `embed_p`, `embed_op`
and all four `forward_*` methods keep their signatures and parameter names, only the declared
type of `phon_*_input` changes from `list[list[torch.Tensor]]` to `torch.Tensor`.
**`training_pipeline.py` needs no edit at all.**

`position_offset` is unused in phase 3 (all callers pass 0) and is what phase 4 needs to stop
re-embedding the prefix. Add it now to avoid a second signature change.

### 6.4 Checkpoint compatibility

**No parameter shape changes.** `phonology_embedding` stays `(36, d_model)`,
`phon_position_embedding` stays `(30, d_model)`. `phon_feature_matrix` is registered
`persistent=False`, so it never enters `state_dict()`, existing `.pth` files load under the
default `strict=True` with no migration.

The residual risk is that `phonreps.csv` and a checkpoint could drift apart. That risk
**already exists on main** (today the tokenizer emits raw feature indices derived from the same
CSV); C relocates it rather than creating it. Make it detectable: add
`phon_table_fingerprint: str | None = None` to `VocabSpec`, populated by `from_tokenizer`.
`ModelConfig` is already saved into checkpoints at
[training_pipeline.py:440](../bridge/application/training/training_pipeline.py#L440), the field
is optional with a default so old checkpoints still deserialize, and `Model.__init__` warns
loudly on mismatch. This also serves the concern behind issue #161.

### 6.5 Generation loop

`phono_sample` already computes `feature_presence: (B, V)`, the dense multi-hot. Keep a
running `(B, step, V)` buffer, extended by `torch.cat` each step.

**Semantics that must be preserved exactly.** In the current
[`phono_sample`](../bridge/domain/model/model.py#L350), an all-off sampled vector becomes
`[PAD]` in `active_features` (used for the *embedding*) but `feature_presence` is returned
**unmodified** (all-zero) and that is what `phon_vecs` records. The two differ. A single
buffer would silently change `phon_vecs`. Keep both:

```python
presence = feature_presence                      # raw -> phon_vecs, unmodified
emb_in = presence.clone()
emb_in[presence.sum(1) == 0, pad_id] = 1         # PAD substitution -> embedding + phon_tokens
```

Then:
- BOS bootstrap: `emb_in` starts as one-hot(`phon_bos_id`), shape `(B, 1, V)`.
- EOS (finding 5): `finished |= presence[:, phon_eos_id].bool()`; `if finished.all(): break`
 , one sync per step instead of an `O(B·L²)` rescan.
- Delete the per-step `for b in range(batch_size)` block at
  [model.py:517](../bridge/domain/model/model.py#L517) (finding 4). Accumulate per-step
  `(B, …)` tensors; after the loop `torch.stack` and split into the `list[list[Tensor]]` shape
  `GenerationOutput` requires.
- Ragged `phon_tokens[b][i] = torch.nonzero(emb_in_buffer[b, i]).squeeze(-1)`, built once at
  the end. `nonzero` returns ascending int64 indices, matching what `torch.where` +
  `torch.tensor(list)` produces today.

`GenerationOutput` is **not changed**; its ragged `phon_tokens` / `phon_probs` / `phon_vecs`
are load-bearing for ~40 assertion sites in `test_generate.py` and for
`phoneme_vectors_to_word`.

### 6.6 Finding 6: share the tokenizer

```python
# BridgeDataset.__init__
def __init__(self, dataset_config, gcs_client=None, tokenizer: BridgeTokenizer | None = None):
    self.tokenizer = tokenizer or BridgeTokenizer(...)
```

```python
# TrainingPipeline.__init__ line 45
self.test_dataset = BridgeDataset(
    dataset_config=test_dataset_config,
    gcs_client=self.dataset.gcs_client,
    tokenizer=self.dataset.tokenizer,      # <- was building a second one
)
```

~5 lines. Saves 993 ms + 81 MB at startup and takes the periodic `gc.collect()` from 188 ms
back to 109 ms. The tokenizer is stateless across datasets apart from its LRU
`vector_cache`, which sharing only helps.

---

## 7. Test plan: written and green **before** implementation starts

This is a large refactor of a numerically sensitive hot path. The tests below are the
acceptance instrument; **all of §7.1–§7.3 land on `main`, passing against the current
implementation, before any production code changes.** That ordering is what makes the rest
safe: every subsequent phase is judged against artefacts captured from behaviour that is known
good.

### 7.1 Tier 0: characterization capture (the safety net)

New: `tests/fixtures/capture_phon_baseline.py` + committed `phon_baseline.pt`.

Run **once on `5d7e8d9`**, before any change. For a fixed word list (≥64 words spanning:
short/long, multi-word phrases with `[SPC]`, an out-of-lexicon word routed through `[UNK]`,
and a cross-language pair) and `ModelConfig(seed=11)`, capture:

| Key | Content |
|---|---|
| `enc_pad_mask`, `dec_pad_mask` | tokenizer masks |
| `targets` | the `(B, L, V-1)` target tensor |
| `phon_feature_sets` | ragged inputs canonicalized to `list[list[sorted tuple[int]]]`, representation-independent, so it survives the ragged→row-id change |
| `embed_out` | `embed_phon_tokens` output |
| `grad_phon_emb`, `grad_pos_emb` | gradients of both tables w.r.t. `embed_out.sum()` |
| `logits_<pathway>` | full forward logits, each of o2p / p2o / p2p / op2op |
| `gen_<pathway>` | `generate(..., deterministic=True)` output: `phon_tokens`, `phon_probs`, `phon_vecs`, `orth_tokens` |

New `tests/domain/test_phon_baseline_equivalence.py` asserts every key. Tolerances:
`torch.equal` for masks/targets/`phon_feature_sets`/`phon_tokens`; `atol=1e-6` for
embeddings and logits; `atol=1e-6, rtol=1e-5` for gradients (measured worst case is 3.8e-6
absolute, which passes on `rtol`, §8).

The `phon_feature_sets` canonicalization is the key trick: it asserts *which features each
position carries*, not how they are stored, so the same assertion holds before and after.

### 7.2 Tier 1: semantics tests (new, written against current behaviour)

| File | Test |
|---|---|
| `tests/core/test_phoneme_table.py` | Row order is `phonreps.csv` order then `SPECIAL_TOKENS`; every phonreps row's multi-hot equals `phonreps_array[i] == 1`; each special token's row is one-hot at its `special_token_dims` index; `normalized` rows sum to 1.0 except the zero-feature row; `fingerprint` is stable across loads and changes when the CSV changes |
| `tests/core/test_phoneme_table.py` | For every phoneme, table row ⇔ what `_get_phoneme_indices` returns today (the bridge between old and new representation) |
| `tests/domain/tokenizer/test_phoneme_tokenizer_targets.py` | `targets` from the gather equals `targets` from the current `torch.isin` loop **bitwise**, including padded positions holding `pad_id` as `ignore_index` (verified in this audit: `torch.equal → True`) |
| `tests/domain/model/test_embed_phon_tokens.py` | Existing 12 tests kept; assertions unedited, inputs converted to row ids. `test_gradient_magnitude_reflects_the_mean_divisor` and `test_ragged_counts_do_not_leak_across_positions` are the load-bearing ones |
| `tests/domain/model/test_embed_phon_tokens.py` | **new:** `embed_phon_tokens(rows)` ≡ `embed_phon_vectors(one_hot(rows))` ≡ explicit `mean(W[features])` reference, `atol=1e-6` |
| `tests/domain/model/test_embed_phon_tokens.py` | **new:** zero-feature phoneme `'_'` → position embedding only, **no `NaN`** (§8.1) |
| `tests/domain/model/test_embed_phon_tokens.py` | **new:** `position_offset=k` equals slicing row `k` of the position table |
| `tests/domain/model/test_generate.py` | **new:** an all-off sampled vector embeds as `[PAD]` **and** `phon_vecs` records the raw all-zero vector (§6.5's asymmetry) |
| `tests/domain/model/test_validate_generate_input.py` | Bounds check is now `rows < P` (=91), **not** `< phon_vocab_size` (=36), a naive port would wrongly reject valid rows 36–90. The existing `TypeError` vs `ValueError` asymmetry must survive |
| `tests/domain/data/test_bridge_dataset.py` | **new:** `BridgeDataset(..., tokenizer=tk)` reuses the instance (`ds.tokenizer is tk`) and produces encodings identical to the un-injected path |

### 7.3 Tier 2: performance guards

`benchmarks/bench_phon.py` (promoted from this audit's scratch harnesses), covering: isolated
`embed_phon_tokens`, `PhonemeTokenizer.encode`, full train step, `generate()`, CPU and CUDA,
across `B ∈ {32, 256, 1024}`.

One `@pytest.mark.perf` (deselected by default) test asserting `embed_phon_tokens` at
`B=256, L=25` is **> 20× faster than a reference double-loop implementation** on CPU. Loose
enough not to flake on shared CI, tight enough to catch a reintroduced Python loop.

### 7.4 Acceptance gates

- [ ] `uv run pytest` green; `uv run mypy bridge` and `uv run ruff check` clean
- [ ] Tier 0 equivalence green for all four pathways
- [ ] `embed_phon_tokens` and `PhonemeTokenizer.encode` contain no Python loop over
      batch/position
- [ ] Measured: isolated `embed_phon_tokens` > 20× (targeting ~220× CPU / ~170× CUDA at
      `B=32`); `encode` > 5× (targeting 10–15× CPU / 31–61× CUDA)
- [ ] Measured: train step ms/step drops; `generate()` drops; both recorded in the PR
- [ ] `EncodingComponent.enc_input_ids` / `dec_input_ids` are `torch.Tensor`, no `Any`
- [ ] An existing pre-change `.pth` checkpoint loads with `strict=True`

---

## 8. Behaviour changes

### 8.1 Zero-feature phoneme: `NaN` → zero vector

`phonreps.csv` contains one phoneme with no active features: `'_'` (86 phonemes, 31 base
features, max 7 active, mean 4.29). Today `phonology_embedding(tokes).mean(axis=0)` over an
empty index tensor yields **`NaN`**; the normalized table yields a zero row → zero vector.

`'_'` is absent from the bundled `en`/`es` lexicons but reachable via `custom_cmudict_path`.
The new behaviour is strictly better. Pin it with a test (§7.2) so it is deliberate.

### 8.2 Float reduction order

Measured against the current implementation:

| | forward max abs diff | grad max abs diff |
|---|---|---|
| CPU | **4.77e-07** | 3.82e-06 |
| CUDA | **2.38e-07** | 3.82e-06 |

Forward is inside `atol=1e-6`. Gradients exceed `atol=1e-6` in absolute terms but pass
`torch.allclose` on the default `rtol=1e-5`, verified `True` on both devices. Gradient
assertions must therefore use `torch.allclose(..., atol=1e-6, rtol=1e-5)`, not a bare
`atol`. **Unlike candidate A, C is not bitwise-identical on CPU** (A is). That is the one
metric on which A wins, and it is not worth 20–90× at scale, but the tolerance choice above
must be explicit rather than discovered during implementation.

### 8.3 Not changed

`GenerationOutput` (still ragged), `phon_targets` semantics, all parameter shapes, the
`nn.Embedding` feature-sharing structure, and every public method signature on `Model`.

---

## 9. Implementation phases

Each phase is a commit that leaves the suite green.

| Phase | Content | Risk |
|---|---|---|
| **0** | §7.1–§7.3 tests, capture fixture on `5d7e8d9`. **No production code.** | none |
| **1** | `PhonemeTable` in `bridge/core/phonreps.py` + §7.2 table tests. Tokenizer injection into `BridgeDataset` and `TrainingPipeline` (**finding 6**). Both are additive and independently valuable. | low |
| **2** | `PhonemeTokenizer.encode` emits row ids + gathered `targets` (**finding 7**). `EncodingComponent` phonological fields become `torch.Tensor`; validation collapses into the orthographic path. `_create_placeholder_phonological` emits `full((B,1), pad_row)`. Model still consumes ragged via a temporary adapter so the suite stays green mid-phase. | medium, the 120 test sites land here |
| **3** | `Model.embed_phon_tokens` / `embed_phon_vectors` per §6.3; `phon_feature_matrix` buffer; `VocabSpec.phon_table_fingerprint`; delete the temporary adapter (**issue #221 proper**). | medium |
| **4** | Generation loop per §6.5: dense buffer, incremental embedding with `position_offset`, dense EOS check, ragged output built once (**findings 1, 4, 5**). | high, gated by the Tier 0 `gen_*` fixtures |
| **5** | Same treatment for `orthography_decoder_loop`'s per-item `for b` loop and `sequence_finished[b]` indexing, the orthographic twin of finding 4. | low |

Phase 4 is the highest-risk step and is exactly where the Tier 0 generation fixtures earn
their keep. Do not start it until phases 0–3 are merged and `gen_*` still matches.

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Row-order contract drifts between tokenizer and model | Single definition in `bridge/core/phonreps.py`; `fingerprint` recorded in `VocabSpec` and checked at `Model.__init__` (§6.4) |
| `phono_sample`'s all-off asymmetry collapsed into one buffer, silently changing `phon_vecs` | Called out in §6.5; dedicated test in §7.2 |
| Bounds check ported as `rows < phon_vocab_size` (36) instead of `< P` (91) | Dedicated test in §7.2; would reject ~60% of valid phonemes, so it fails loudly |
| Divisor bug (dividing by padded width, not true count) | Impossible by construction in C, `A_norm` is normalized once at load, and `test_gradient_magnitude_reflects_the_mean_divisor` still guards it |
| Gradient assertions written with bare `atol=1e-6` fail at 3.8e-6 | §8.2 fixes the tolerance up front |
| Generation output drifts in phase 4 | Tier 0 `gen_*` fixtures captured before phase 1, asserted every phase |
| Per-call `A_norm @ W` GEMM adds a kernel launch on CUDA | Measured: C is within noise of B on CUDA at every shape and fastest on CPU everywhere. If it ever matters, cache the table per `forward()`, a local variable threaded through, not new API |
| 120 test sites is a large diff | Phase 2 isolates it; the adapter keeps the suite green mid-phase; the diff is mostly ragged→tensor substitution |

---

## 11. Out of scope

- **A KV cache for the decoders.** Phase 4 removes the `O(L²)` *re-embedding*, but both
  decoder loops still recompute full attention over the prefix every step. That is the next
  order-of-magnitude win in `generate()` and is a much larger change.
- **`d_embedding` / position-encoding work** (issues #101, #83), orthogonal.
- **Resaving existing `.pth` files** (issue #161). Phase 3's fingerprint field makes future
  checkpoints self-describing; back-filling old ones is separate.
