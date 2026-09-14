# 0006. The caller owns the training loop; the library owns the step

Status: Accepted
Date: 2026-08-30

## Context

Issue #224 defect 5 reported that `TrainingPipeline.save_model` accepted a `run_name`, was
passed one at its only call site, and never used it. The filename depended on the epoch
alone, so two runs sharing a `model_artifacts_dir` silently overwrote each other. Measured:
`save_model(epoch=0, run_name="run_a")` followed by `save_model(epoch=0, run_name="run_b")`
left exactly one file, `model_epoch_0.pth`.

The issue prescribed a fix: put the run name in the filename. Two layouts were defensible.
A flat prefix, `<dir>/<run_name>_epoch_<n>.pth`, as the issue suggested. Or a per-run
subdirectory, `<dir>/<run_name>/model_epoch_<n>.pth`, which `bridge/utils/helper_functions.py`
already implies: `get_run_name` creates exactly that directory and nothing ever writes into
it.

Choosing between them settles a question the library should not be answering. BRIDGE is
increasingly imported and driven from experiment code rather than operated from inside this
repository, and no naming scheme picked here is right for every experiment.

## Decision

The library owns the step. The caller owns the loop. The default loop is a thin wrapper over
the public step. This is the shape Lightning and torchtune converged on.

Concretely:

- **`save_checkpoint(path, epoch)`** replaces `save_model(epoch, run_name)`. It takes a
  destination and writes there. A relative path resolves against `model_artifacts_dir` so
  the short form stays short; parent directories are created, which is the point of first
  write now that validating a config no longer touches the filesystem. The bundle it writes
  is unchanged, because *what* belongs in a checkpoint is library knowledge even when *when*
  and *where* are not.

- **`save_every` is deleted** from `TrainingConfig`. A cadence is save policy, and policy
  moved out with the naming.

- **`single_step` and `train_steps(epoch)` are supported API.** `train_steps` runs one epoch
  and yields after every optimizer step. It deliberately does not shuffle: reordering belongs
  to the epoch, and a caller driving it across several epochs should not get a permutation it
  did not ask for.

- **`run_train_val_loop(num_epochs=None)`** yields a `TrainingEvent` per step and per
  boundary, and writes no checkpoints. `run_name` is gone from its signature, having nothing
  left to name.

`TrainingEvent` carries `phase`, `epoch`, `step` and `metrics`. Four phases arrive:

| phase | when | `step` |
|---|---|---|
| `train` | one optimizer step | its index in the epoch |
| `validation` | the validation partition, per epoch | `None` |
| `test` | the held-out test set, per epoch, when configured | `None` |
| `epoch` | the epoch summary, always last for its epoch | `None` |

## Why the epoch event exists

Yielding only per step would have been simpler, and it was rejected. Validation is a
per-epoch operation, so a per-step stream has to surface epoch boundaries somehow or the
caller reconstructs them by watching `step` reset, which is a contract nobody wrote down.

The `epoch` event carries the merged aggregate this generator yielded before it yielded per
step. A caller who only wants epoch rows filters on that phase and is otherwise unchanged,
which is what makes the granularity change additive rather than a rewrite for every consumer.

`TrainingEvent` is a frozen dataclass rather than a pydantic model, unlike the configs and
`GenerationOutput`. Those sit on validation boundaries where a caller supplies the values.
This is a carrier the pipeline fills in itself, once per optimizer step, so per-field
validation would cost something on the hot path and defend against nothing. The one
invariant worth enforcing, that `step` is present exactly when the phase is `train`, is a
`__post_init__` check.

## Consequences

`run_train_val_loop` is a breaking change for anything outside this repository that consumes
it: the signature lost `run_name`, the granularity changed, and it no longer checkpoints. The
migration is mechanical:

```python
# before
for metrics in pipeline.run_train_val_loop(run_name):
    ...

# after
for event in pipeline.run_train_val_loop():
    if event.phase != "epoch":
        continue
    metrics = event.metrics
    pipeline.save_checkpoint(f"{run_name}/model_epoch_{event.epoch}.pth", event.epoch)
```

A caller who forgets the last line trains a model and saves nothing. That is the cost of
moving the policy out, and it is deliberate: silently overwriting another run's weights was
the worse failure, because it looked like success.

`metrics_logger.save()` used to be called from inside `save_model`, which tied flushing the
metrics to the checkpoint cadence for no reason. It now runs when the loop finishes.

`get_run_name` still creates a per-run directory and still nothing writes into it. That is
now a caller's convenience rather than a half-built convention, since `save_checkpoint`
accepts the path it implies.
