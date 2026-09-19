# 0012. The training pipeline follows the model's device, it does not set it

Status: Accepted
Date: 2026-09-16

## Context

`docs/decisions/0007` established that `Model.device` is derived from a parameter so `.to()`
is authoritative and the module cannot misreport where it lives. `TrainingPipeline.__init__`
then did this:

```python
self.device = device_manager.device
self.model = model.to(self.device)
```

Both mistakes 0007 removed, one layer up: a stored device snapshot, and a placement decision
taken away from the caller.

0007 kept the `model.to(...)` call deliberately, describing it as "redundant rather than
load-bearing" for a model built before the manager moved. That reasoning does not cover a
model the caller placed on purpose. Measured at commit `7979d5e`, CUDA available,
`BRIDGE_DEVICE` unset so the manager resolved to CPU:

```
after construction        model.device = cpu
after caller model.to(cuda) model.device = cuda:0    169/169 params on cuda:0
after TrainingPipeline(..)  model.device = cpu       169/169 params on cpu
```

`model.to("cuda"); TrainingPipeline(model, ...)` trains on CPU. Nothing raises. The only
signal is an INFO log line, and the symptom is a run an order of magnitude slower than
expected, which is exactly the failure `bridge/utils/device_manager.py` already warns about
in `set_device`'s docstring.

## Decision

**`TrainingPipeline.device` is a property returning `self.model.device`.** The pipeline
asks the model where it is. It does not move it, and it stores nothing.

A caller who wants the process device writes `model.to(device_manager.device)` before
constructing the pipeline. That is one line, at the call site, where it is visible.

## Consequences

Placement has one owner, the caller, and one witness, the model's parameters.

`BRIDGE_DEVICE=cuda` still works for a caller who does not place the model themselves,
because `Model.__init__` reads `device_manager.device` and places the module once at the end
of construction, which 0007 already settled.

Four constructors still snapshot `device_manager.device` into `self.device`:
`BridgeTokenizer`, `CharacterTokenizer`, `PhonemeTokenizer` and `BridgeDataset`. They are
not covered by this record. Tokenizers build tensors rather than hold parameters, so the
same argument does not transfer unchanged, and `BridgeEncoding.to()` already exists for the
consumer to place the result. `tests/conftest.py` still pops `BRIDGE_DEVICE` before the
manager is first imported, for the reason recorded there.
