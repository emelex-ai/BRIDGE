# 0010. The library computes; the caller does the I/O

Status: Accepted
Date: 2026-09-16

## Context

`bridge/infra/` held four things: a Google Cloud Storage client, a Weights and Biases
wrapper, a `MetricsLogger` hierarchy writing CSV to a `results/` directory, and a
`StorageInterface` reading `os.environ["CLOUD_RUN_TASK_INDEX"]` at construction. None of
it is model code. All of it was reachable from the import graph of the core:

```
application  -> infra   training_pipeline.py -> bridge.infra.metrics.metrics_logger
domain       -> infra   bridge_dataset.py    -> bridge.infra.clients.gcp.gcs_client
infra        -> application  wandb_wrapper.py -> bridge.application.shared.Singleton
```

That last edge is a cycle: a nine-line `Singleton` in the nominally highest layer existed
to serve one class in the nominally lowest.

The cost was not the 430 lines. `TrainingPipeline.__init__` **required** a concrete
`MetricsLogger` with no default, so:

- `metrics_logger_factory` had to be re-exported from `bridge/__init__.py` purely to let a
  consumer satisfy the constructor, and `tests/test_public_api.py` carried a test arguing
  that it should be;
- six test modules reached into `bridge.infra.metrics.metrics_logger` for
  `STDOutMetricsLogger` to build a pipeline at all;
- `google-cloud-storage` was a hard install dependency of a model library, pulled in
  unconditionally by `import bridge`;
- `wandb` was imported at module scope in `bridge/`, but declared only in the dev
  dependency group, so the import the README instructed consumers to write
  (`from bridge.infra.clients.wandb import WandbWrapper`) raised `ModuleNotFoundError`
  after a plain install.

The subsystem was also barely used. Every `MetricsConfig` constructed anywhere in the repo,
24 of them, set `batch_metrics`, `training_metrics` and `validation_metrics` to `False` and
`modes` to `[]`. A line tracer over the full suite recorded zero executed lines in
`CSVMetricsLogger`, `CSVGCPMetricsLogger` and both `MultipleMetricsLogger` methods.

Meanwhile `docs/decisions/0006` had already established the seam that makes a logger
unnecessary: `run_train_val_loop` emits a `TrainingEvent` per step and per boundary, and
the caller decides what to do with each.

## Decision

**Delete `bridge/infra/` and `bridge/application/shared/`.** The library performs no
network I/O, opens no files it was not handed a path to, reads no environment variables for
destinations, and prints nothing.

**`TrainingPipeline(model, training_config, dataset)`.** Three arguments. Whether to score
metrics is `TrainingConfig.compute_metrics`; where the numbers go is the caller's, from the
event stream.

**`save_checkpoint` writes one file and returns its path.** It no longer uploads to a bucket
named by `os.environ["BUCKET_NAME"]`, which could turn a successful `torch.save` into a
`KeyError` on the next line.

**`BridgeDataset` accepts any object with `read_csv(bucket_name, blob_name, **kwargs)`,**
declared as a `CSVReader` protocol. `gs://` dataset paths still work; the client comes from
the caller and no cloud SDK is installed.

## Consequences

The runtime dependency list is torch, pydantic, pandas, numpy and tqdm. `pyyaml`,
`torchsummary`, `protobuf` and `google-cloud-storage` are gone.

The layer cycle is gone with the modules that formed it.

A downstream repo that used `metrics_logger_factory` must write its own sink. Against the
event stream that is a handful of lines, and it was already going to need one, because the
CSV logger derived its filename by splitting on `"."` and so could not name a file
`run.2026.csv`.

`bridge/__init__.py` loses `MetricsConfig` and `metrics_logger_factory` and gains
`PhonemeTable`, `load_phoneme_table`, `PATHWAYS`, `Pathway`, `CharacterTokenizer` and
`PhonemeTokenizer`. `docs/architecture.md` names `features_of` and `row_of` as the remedy
for the two-id-space hazard, and neither was reachable through a supported import.

W&B integration is not replaced. A caller writes `wandb.log(event.metrics)` inside their own
loop, which is shorter than the wrapper was and does not need a singleton.
