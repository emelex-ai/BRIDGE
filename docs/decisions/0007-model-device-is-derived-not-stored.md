# 0007. A model's device is derived from its parameters, never stored

Status: Accepted
Date: 2026-09-14

## Context

`Model.__init__` recorded `self.device = device_manager.device` and then created exactly one
of its 169 parameters there:

```python
self.global_embedding = nn.Parameter(
    torch.randn((1, d_embedding, d_model), device=self.device) / d_model**0.5,
)
```

Every `nn.Embedding`, `nn.Linear` and transformer layer was built without a `device`
argument, so they took torch's default. Measured on an RTX 5080 with the manager pointed at
CUDA: parameters split `['cpu', 'cuda:0']`, 1 on the GPU and 168 on the CPU, against a
reported device of `cuda:0`. Every generation pathway then raised
`Expected all tensors to be on the same device`.

`nn.Module.to` knows nothing about a plain attribute, so the stored value also did not follow
the module. A model moved anywhere other than `device_manager.device` kept building its
causal masks and generation buffers, all of which use `self.device`, on the device it used to
be on. That is #223 from the opposite direction: the same class of silently wrong device
comparison, arrived at by moving the model rather than by mis-resolving the manager.

It stayed invisible because `TrainingPipeline.__init__` calls `model.to(self.device)`, which
repairs the placement immediately, and because `device_manager` had no supported way to
select a GPU until #224 defect 7 was fixed. A plain `Model(cfg)` on a GPU host only became a
reachable state once it did.

## Decision

**Place the module once, at the end of `__init__`,** with `self.to(target_device)` where
`target_device` is a local rather than an attribute. Per-submodule placement is what allowed
168 parameters to be missed, and a submodule added later would be missed the same way.

**Derive `device` from a parameter** rather than storing it:

```python
@property
def device(self) -> torch.device:
    return self.global_embedding.device
```

`.to()` is then authoritative, which is the contract every other PyTorch module has. The
property is read-only, so the report cannot be set out of sync with the module.

`global_embedding` is the witness only because it is a plain parameter every configuration
has. The module is placed as a whole, so any parameter would answer the same.

## Evidence

**CPU behaviour is unchanged, bitwise.** Drawing `global_embedding` on the default device
rather than on an explicitly-named CPU is the same draw, and `self.to("cpu")` is a no-op. A
differential over 4 seeds by 2 shapes, comparing all 169 parameters and the 1 buffer with
`torch.equal`:

| configurations | tensors compared | bitwise identical |
|---|---|---|
| 8 | 1360 | **1360** |

That matters because `tests/fixtures/phon_baseline.pt` and every recorded loss trajectory
were taken on CPU. Script: `fingerprint_model.py`.

**On CUDA the model is now whole.** 169 parameters and 1 buffer all on `cuda:0`, reported as
`cuda:0`, and all five generation pathways run with no `.to()` from the caller.

**Initialisation stopped depending on where the model runs.** Because the weights are drawn
on the default device and then moved, one generator produces them and a seed means the same
thing everywhere. Measured, seed 5, `d_model=32`:

| comparison | parameters bitwise identical |
|---|---|
| CPU build vs CUDA build | **169 / 169** |

Before the change `global_embedding` alone came from the CUDA generator, so that one
parameter differed between a CPU and a GPU run of the same seed. Nothing depended on the old
behaviour, and device-independent initialisation is worth more than matching it.

## Consequences

`Model.device` is read-only. Code that assigned it would now raise, which is the point; a
search of `bridge/` and `tests/` at the time of the change found no such assignment.

Placement happens once, so a `Model` is usable immediately after construction.
`TrainingPipeline`'s `model.to(self.device)` is now redundant rather than load-bearing. It is
kept: a pipeline handed a model built before the manager moved should still place it, and the
call is idempotent.

The `device=` argument is gone from the `load_phoneme_table` call in `__init__`. The table is
cached per device, so this builds the CPU copy and moves it with everything else rather than
populating a second cache entry.
