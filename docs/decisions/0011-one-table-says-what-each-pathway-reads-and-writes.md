# 0011. One table says what each pathway reads and writes

Status: Accepted
Date: 2026-09-16

## Context

Five pathways, and thirteen places that each answered "which of them use orthography" with
their own literal tuple:

```python
model.py:31              ORTH_DECODING = ("op2op", "p2o", "o2o")
model.py:826             if pathway in ["op2op", "o2p", "p2p"]:
model.py:907             uses_orth = pathway in ("o2p", "o2o", "op2op")
training_pipeline.py:144 if ... training_pathway in ["p2o", "op2op"]:
training_pipeline.py:213 if ... training_pathway in ["op2op", "p2o"]:
training_config.py:70    allowed_training_pathways = ["o2p", "p2o", "op2op", "p2p"]
```

The last two `training_pipeline` lines are the same set in different orders, one gating the
loss and the other the metrics, 69 lines apart. A metric scoring positions the loss never
trains is the exact shape of issue #225, and nothing made the two agree.

`training_config.py` held a sixth spelling: `PATHWAYS` minus `o2o`, with nothing recording
the relationship. `o2o` was declared in `PATHWAYS`, accepted by `Model.generate`, and absent
from `Model.forward`'s dispatch dict, so it generated but could not train and said so with
`ValueError("Invalid pathway selected.")`, which names neither the pathway nor the valid set.

`TrainingPipeline.forward` spelled the tensor mapping out per pathway, four branches of
six-to-ten keyword arguments. The same mapping appears again in
`tests/fixtures/capture_phon_baseline.py`, which records the golden master, so renaming a
keyword in one place left the baseline recording the old convention.

`Model._validate_generate_input` was five branches applying four different subsets of the
available checks:

| pathway | structural | length bound | id bounds | device | non-tensor |
|---|---|---|---|---|---|
| `p2o`, `p2p` | phon | yes | yes | mask only | `ValueError` |
| `o2p` | orth | no | no | no | `ValueError` |
| `o2o` | orth | no | yes | yes | `ValueError` |
| `op2op` | both | yes | yes | no | `TypeError` |

`o2p` is the default `training_pathway` and the least validated of the five. None of those
differences was designed; they are what a partial refactor left, and a 404-line test had
since pinned them in place, including the `TypeError`, which its own docstring described as
predating the validator. The gaps are issue #233.

## Decision

**`PATHWAY_IO` in `bridge/domain/datamodels/pathways.py` is the single definition.** Each pathway
maps to the set of modalities it reads and the set it writes. Everything else is a query:

```python
READS_ORTH  = tuple(p for p in PATHWAYS if "orth" in PATHWAY_IO[p][0])
WRITES_PHON = tuple(p for p in PATHWAYS if "phon" in PATHWAY_IO[p][1])
PATHWAY_INPUTS = {...}   # read -> encoder input, written -> decoder input
```

`PATHWAY_INPUTS` is derived rather than written: a modality that is read supplies encoder
tensors and one that is written supplies decoder tensors, which is the whole rule.

**Validation runs one check set on whatever the pathway reads.** Structure, dtype, mask
shape, length bound, id range and device placement, for every pathway, plus a zero-row
rejection. `TypeError` for a non-tensor is gone; everything is `ValueError`.

**`forward_o2o` exists,** so the five declared pathways are five working pathways and
`TrainingConfig.training_pathway` is `Pathway` rather than a hand-maintained subset.

It lives in `datamodels` rather than in `model.py`, and that placement is load-bearing
rather than tidy. With the table inside the model, `TrainingConfig` importing `Pathway`
closed a cycle: `datamodels/__init__` imports `training_config`, which imports
`bridge.domain.model.model`, which imports back from the partially initialised
`bridge.domain.datamodels`. It resolved only because `model.py` happened to need three
names that `__init__` binds before line 5. Adding a fourth to that same import line, one
word, gave `ImportError: cannot import name 'VocabSpec' from partially initialized module`
on a bare `import bridge`. A pathway table is data about modality structure, which is what
`datamodels` is for, and from there nothing imports upward. `model.py` re-exports every
name, so existing imports are unaffected.

## Consequences

`o2p` now rejects out-of-range ids and wrong-device tensors, which it always should have.
That is a behaviour change, and it is the point.

**Correction, 2026-09-19.** This record originally claimed that two of issue #233's three
shapes were "rejected at the boundary now". Only one is. The zero-row batch that segfaulted
the CUDA decoder is rejected, verified. The phonology-only encoding is **not**: measured on
all three orthographic pathways, `tok.encode(words, modality_filter="phonology")` followed
by `model.generate(enc, "o2p")` still raises
`RuntimeError: to_padded_tensor: at least one constituent tensor should have non-zero numel`
from inside the encoder, byte-identically to the behaviour before this change.

The new absence check cannot fire on it, and the reason is worth recording. `Model.generate`
passes `None` for any modality the pathway does not read, so the check only ever sees
arguments `generate` itself constructed. What actually reaches `embed_o` is the two-wide
`[--, BOS]` placeholder that `BridgeTokenizer._create_placeholder_orthographic` puts on
every `BridgeEncoding`, and a placeholder is structurally indistinguishable from a real
two-character encoding. Closing it needs the encoding to record which modalities are real,
not a better check at the boundary. Tracked under #233, which stays open on that shape.

`tests/domain/model/test_validate_generate_input.py` drops from 404 lines to 190 and covers
more: it sweeps every pathway against every modality it reads against every malformation,
so a pathway added to `PATHWAY_IO` is covered the moment it is added, rather than needing
its own hand-written block.

Adding a sixth pathway means adding a row to `PATHWAY_IO` and a `forward_*` method. It no
longer means finding thirteen tuples.

The remaining `[:-4]`-style literals about modality structure are gone from the training
code, but `tests/fixtures/capture_phon_baseline.py` still spells its own mapping, because
the golden master must be reproducible from the code as it was when recorded.
