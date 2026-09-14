"""Checkpointing is the caller's decision, and the pipeline's format.

Two training runs pointed at the same artifacts directory used to overwrite each other
without a word: ``save_model`` took a ``run_name``, was passed one, and built its filename
from the epoch alone. The repair is not a better naming scheme. It is that no naming scheme
the library picks can be right for every experiment, so ``save_checkpoint`` takes a path and
``save_every`` is gone. See docs/decisions/0006-the-caller-owns-the-training-loop.md.

The oracle throughout is a round trip: what the pipeline is told to record has to come back
off disk saying the same thing. Where the file lands is checked against paths computed in
the test rather than read back from the pipeline, so the assertions cannot agree with the
code by construction.
"""

import ast
import inspect

import pytest
import torch

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.infra.metrics.metrics_logger import STDOutMetricsLogger

DATA_CSV = "tests/domain/model/data/data.csv"

SILENT_METRICS = MetricsConfig(
    batch_metrics=False,
    training_metrics=False,
    validation_metrics=False,
    modes=[],
    filename=None,
)


@pytest.fixture(scope="module")
def dataset():
    """Parsing the 7300-word csv is the slow part, so it happens once."""
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA_CSV))


def make_pipeline(dataset, artifacts_dir, **overrides):
    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
    return TrainingPipeline(
        model=Model(ModelConfig(vocab=vocab, d_model=16, nhead=2, seed=5)),
        dataset=dataset,
        training_config=TrainingConfig(
            num_epochs=1,
            training_pathway="o2p",
            model_artifacts_dir=str(artifacts_dir),
            **overrides,
        ),
        metrics_logger=STDOutMetricsLogger(SILENT_METRICS),
    )


def test_distinct_paths_keep_distinct_checkpoints(dataset, tmp_path):
    """The defect, stated as the invariant it violated.

    Two runs sharing an artifacts directory previously collapsed to a single
    ``model_epoch_0.pth``. Injectivity is asserted here by counting distinct files against
    distinct requests, rather than by pinning filename literals, so any layout the caller
    chooses satisfies it: a prefix, a subdirectory, a timestamp.
    """
    pipeline = make_pipeline(dataset, tmp_path)

    requests = [
        ("run_a/model_epoch_0.pth", 0),
        ("run_b/model_epoch_0.pth", 0),
        ("run_a/model_epoch_1.pth", 1),
        ("flat_run_c_epoch_0.pth", 0),
    ]
    written = [pipeline.save_checkpoint(path, epoch) for path, epoch in requests]

    assert len({str(p) for p in written}) == len(requests), (
        f"distinct requests collapsed onto the same path: {written}"
    )
    on_disk = sorted(p for p in tmp_path.rglob("*.pth"))
    assert len(on_disk) == len(requests), f"expected {len(requests)} files, found {on_disk}"
    for destination in written:
        assert destination.exists()


def test_a_relative_path_lands_under_the_artifacts_directory(dataset, tmp_path):
    """Short paths are the common case, so a relative one resolves rather than escaping.

    Asserted against a path this test builds, not against what the pipeline returns
    compared with itself.
    """
    pipeline = make_pipeline(dataset, tmp_path)

    destination = pipeline.save_checkpoint("nested/run/model_epoch_2.pth", epoch=2)

    assert destination == tmp_path / "nested" / "run" / "model_epoch_2.pth"
    assert destination.exists(), "intermediate directories were not created"


def test_an_absolute_path_is_written_exactly_there(dataset, tmp_path):
    """The control for the test above: an absolute path must not be re-rooted."""
    pipeline = make_pipeline(dataset, tmp_path / "artifacts")
    elsewhere = tmp_path / "somewhere_else" / "checkpoint.pth"

    destination = pipeline.save_checkpoint(elsewhere, epoch=0)

    assert destination == elsewhere
    assert elsewhere.exists()
    assert not (tmp_path / "artifacts").exists(), "an absolute path must not touch the default"


def test_the_bundle_round_trips(dataset, tmp_path):
    """What goes in comes back out, including the epoch `load_model` resumes from.

    The state dict is compared against a copy taken before saving rather than against the
    live model, so a save that wrote nothing and a load that returned the in-memory object
    would both fail here.
    """
    pipeline = make_pipeline(dataset, tmp_path)
    before = {k: v.detach().clone() for k, v in pipeline.model.state_dict().items()}

    destination = pipeline.save_checkpoint("model_epoch_7.pth", epoch=7)
    loaded = torch.load(destination, weights_only=False)

    assert loaded["epoch"] == 7
    assert set(loaded["model_state_dict"]) == set(before)
    for key, saved in loaded["model_state_dict"].items():
        assert torch.equal(saved, before[key]), f"{key} did not survive the round trip"
    assert loaded["model_config"].d_model == 16
    assert "optimizer_state_dict" in loaded
    assert loaded["dataset_config"].dataset_filepath == dataset.dataset_config.dataset_filepath


def test_the_gcs_destination_is_named_after_the_file_written(dataset, tmp_path, monkeypatch):
    """The upload follows the local name, so the two cannot disagree.

    The oracle is the argument value the pipeline computes, not that a double was called at
    all. A recorder that only proved a call happened would pass against any destination
    string whatsoever, including the old epoch-only one.
    """
    monkeypatch.setenv("BUCKET_NAME", "a-bucket")
    pipeline = make_pipeline(dataset, tmp_path, gcs_path="experiments/run17")

    uploads = []

    class Recorder:
        def upload_file(self, bucket, local, remote):
            uploads.append((bucket, local, remote))

    monkeypatch.setattr(pipeline.dataset, "gcs_client", Recorder())
    destination = pipeline.save_checkpoint("run_a/model_epoch_4.pth", epoch=4)

    assert uploads == [("a-bucket", str(destination), "experiments/run17/models/model_epoch_4.pth")]


def test_no_gcs_client_means_no_upload(dataset, tmp_path, monkeypatch):
    """The control showing the recorder above can observe a call NOT happening."""
    pipeline = make_pipeline(dataset, tmp_path, gcs_path="experiments/run17")
    monkeypatch.setattr(pipeline.dataset, "gcs_client", None)

    destination = pipeline.save_checkpoint("model_epoch_0.pth", epoch=0)

    assert destination.exists(), "the local write still happens without a GCS client"


def test_the_loop_writes_no_checkpoints_of_its_own(dataset, tmp_path):
    """`run_train_val_loop` saves nothing. That is the whole change.

    With `save_every` gone there is no cadence for the library to apply, so a run that the
    caller never asks to checkpoint must leave the artifacts directory empty.
    """
    pipeline = make_pipeline(dataset, tmp_path)
    pipeline.train_slices = pipeline.train_slices[:1]
    pipeline.val_slices = pipeline.val_slices[:1]

    events = list(pipeline.run_train_val_loop(num_epochs=1))

    assert events, "the loop yielded nothing at all"
    assert list(tmp_path.rglob("*.pth")) == [], "the loop wrote a checkpoint nobody asked for"


def test_save_every_is_gone(tmp_path):
    """A leftover cadence field would quietly reintroduce library-owned save policy.

    Three checks, because one is not enough. The field is off the schema; a config that is
    handed one anyway does not grow the attribute, since pydantic ignores extras by default
    and a caller's old kwargs will keep arriving for a while; and no code reads it. The last
    is the one that matters, and it is a search with a stated space: the two modules that
    ever referred to it.
    """
    assert "save_every" not in TrainingConfig.model_fields

    config = TrainingConfig(model_artifacts_dir=str(tmp_path), save_every=2)
    assert not hasattr(config, "save_every"), "an ignored kwarg still reached the model"

    # The search space: every attribute access and every name bound in the two modules that
    # ever mentioned the field. Parsed rather than grepped, because a substring search over
    # the source matches the docstring above explaining why the field is gone, which is a
    # detector that cannot tell code from prose about code.
    for module in (TrainingPipeline, TrainingConfig):
        tree = ast.parse(inspect.getsource(inspect.getmodule(module)))
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assigned = {
            n.target.id
            for n in ast.walk(tree)
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }
        found = {"save_every"} & (attributes | assigned)
        assert not found, f"{module.__name__}'s module still reads save_every"
