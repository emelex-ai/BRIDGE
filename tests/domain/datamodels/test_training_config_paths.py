"""Pins where ``TrainingConfig`` puts a relative ``model_artifacts_dir``, and when it
touches the disk.

``convert_paths`` currently joins a relative ``model_artifacts_dir`` onto
``get_project_root()``, which is BRIDGE's own install root rather than the directory the
caller is working in, and then creates that directory as a side effect of validation. A
user who passes ``"runs/experiment1"`` therefore gets checkpoints written inside the
installed package, where the next reinstall deletes them. Absolute paths escape only by
accident: ``os.path.join`` discards its prefix when the second argument is absolute.

The contract asserted here is that a relative directory means "relative to the working
directory", the same rule ``open("runs/x")`` follows, and that constructing a config is a
pure function of its arguments: nothing appears on disk until something is actually
written. Creation moves to ``TrainingPipeline.save_model``, so the last test builds a real
pipeline and checks that saving still works once eager creation is gone.

``checkpoint_path`` is the control. It names a file that must already exist, so its
existence check has to survive; if that check disappeared along with the directory check,
a typo in a resume path would fail deep inside ``load_model`` instead of at construction.
"""

import os
from pathlib import Path

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
from bridge.utils import get_project_root

RELATIVE_DIR = "bridge_relative_artifacts_probe"


@pytest.fixture
def no_repo_litter():
    """Remove the directory today's code wrongly creates inside the install root.

    The defect under test is that a relative ``model_artifacts_dir`` is created under
    ``get_project_root()``. Running the red half of this pair therefore leaves an empty
    directory in the checked-out repository. Once the fix lands nothing is created and
    this fixture does nothing.
    """
    yield
    stray = Path(get_project_root()) / RELATIVE_DIR
    if stray.is_dir() and not any(stray.iterdir()):
        stray.rmdir()


# --- where a relative directory lands --------------------------------------


def test_relative_artifacts_dir_resolves_under_the_working_directory(
    tmp_path, monkeypatch, no_repo_litter
):
    """A relative path means "from here", not "from wherever BRIDGE happens to be installed".

    Both halves matter. The directory has to sit under the caller's working directory, and
    it has to *not* sit under the install root, which is where it goes today.
    """
    monkeypatch.chdir(tmp_path)
    cwd = Path.cwd()

    config = TrainingConfig(model_artifacts_dir=RELATIVE_DIR)
    resolved = Path(config.model_artifacts_dir)

    assert resolved.is_absolute()
    assert resolved.name == RELATIVE_DIR
    assert resolved.parent == cwd, (
        f"relative artifacts dir resolved to {resolved}, expected it under the working "
        f"directory {cwd}; the install root is {get_project_root()}"
    )
    assert Path(get_project_root()) not in resolved.parents, (
        f"{resolved} is inside the install root, so a reinstall would delete the checkpoints"
    )


def test_the_default_artifacts_dir_resolves_under_the_working_directory(monkeypatch, tmp_path):
    """The default goes through the same join as an explicit value, so it needs its own test.

    ``convert_paths`` calls ``values.setdefault(...)`` before resolving, so a config
    constructed with no ``model_artifacts_dir`` at all is resolved by exactly the same two
    lines. A fix that only handled explicitly-passed values would leave the default landing
    inside the install root and every test above would still pass.
    """
    monkeypatch.chdir(tmp_path)

    config = TrainingConfig()
    resolved = Path(config.model_artifacts_dir)

    assert resolved == tmp_path / "model_artifacts"
    assert not resolved.is_relative_to(Path(get_project_root())), (
        f"the default landed inside the install root: {resolved}"
    )
    assert not resolved.exists(), "validation must not create the default directory either"


def test_absolute_artifacts_dir_is_preserved_exactly(tmp_path):
    """An absolute path is already an answer, so resolution must be the identity on it.

    This is the round trip: what goes in comes back out unchanged. It works today only
    because ``os.path.join`` drops its prefix for an absolute second argument, so it needs
    stating explicitly rather than relying on that accident.
    """
    given = str(tmp_path / "runs" / "experiment1")

    config = TrainingConfig(model_artifacts_dir=given)

    assert config.model_artifacts_dir == given


# --- validation is pure ----------------------------------------------------


def test_construction_creates_nothing_on_disk(tmp_path):
    """Validating a config must not write to the filesystem.

    Constructing a ``TrainingConfig`` to inspect it, to compare two of them, or to load one
    from a sweep file should not scatter directories around. The planted directory at the
    end is the control: it proves the emptiness check is looking at the right place and
    would notice a directory if one had appeared.
    """
    artifacts = tmp_path / "not_created_by_validation"

    config = TrainingConfig(model_artifacts_dir=str(artifacts))

    assert config.model_artifacts_dir == str(artifacts)
    assert not artifacts.exists(), "validation created the artifacts directory as a side effect"
    assert list(tmp_path.iterdir()) == []

    (tmp_path / "planted").mkdir()
    assert [p.name for p in tmp_path.iterdir()] == ["planted"]


def test_missing_artifacts_dir_is_not_an_error(tmp_path):
    """A directory that does not exist yet is normal, not a failure.

    The old ``validate_paths`` check raised ``FileNotFoundError`` here, which was only ever
    unreachable because the eager ``makedirs`` ran first. With creation deferred to the
    first save, the check has to go or every fresh run would refuse to start.
    """
    artifacts = tmp_path / "fresh_run"

    config = TrainingConfig(model_artifacts_dir=str(artifacts))

    assert config.model_artifacts_dir == str(artifacts)


# --- test_data_path is untouched by this change ----------------------------


def test_relative_test_data_path_still_joins_project_root_data(tmp_path):
    """Golden master of the current ``test_data_path`` rule, so the edit cannot move it.

    ``convert_paths`` treats ``test_data_path`` differently from ``model_artifacts_dir``: a
    truthy value is joined onto ``get_project_root()`` plus a literal ``data`` segment, an
    absolute value survives because ``os.path.join`` discards the prefix, and ``None``
    passes straight through. Nothing about this is validated for existence. The fix targets
    ``model_artifacts_dir`` only, so all three of these must read the same afterwards.
    """
    root_data = os.path.join(get_project_root(), "data")
    artifacts = str(tmp_path / "artifacts")

    relative = TrainingConfig(model_artifacts_dir=artifacts, test_data_path="held_out.csv")
    assert relative.test_data_path == os.path.join(root_data, "held_out.csv")

    absolute_path = str(tmp_path / "held_out.csv")
    absolute = TrainingConfig(model_artifacts_dir=artifacts, test_data_path=absolute_path)
    assert absolute.test_data_path == absolute_path

    unset = TrainingConfig(model_artifacts_dir=artifacts)
    assert unset.test_data_path is None


# --- checkpoint_path is the control ----------------------------------------


def test_missing_checkpoint_raises_at_construction(tmp_path):
    """A checkpoint that is not there is a real error, and it must still fire early.

    This is the half of ``validate_paths`` that stays. It is the control for the directory
    check being removed: if both disappeared, this test would go quiet and a mistyped
    resume path would only surface as a caught exception inside ``load_model``.
    """
    missing = tmp_path / "no_such_checkpoint.pth"

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        TrainingConfig(
            model_artifacts_dir=str(tmp_path / "artifacts"),
            checkpoint_path=str(missing),
        )


def test_existing_checkpoint_is_accepted(tmp_path):
    """The other side of the control: a checkpoint that exists must construct cleanly."""
    checkpoint = tmp_path / "model_epoch_0.pth"
    torch.save({"epoch": 0}, checkpoint)

    config = TrainingConfig(
        model_artifacts_dir=str(tmp_path / "artifacts"),
        checkpoint_path=str(checkpoint),
    )

    assert config.checkpoint_path == str(checkpoint)


# --- saving still works once creation is deferred --------------------------


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One dataset for the module. Building a tokenizer parses the pronunciation lexicons."""
    words = tmp_path_factory.mktemp("data") / "words.csv"
    words.write_text("word_raw,count\nlong,3\npencil,2\ncat,5\ndog,4\nhouse,2\nbook,7\n")
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=str(words)))


def test_save_model_creates_the_artifacts_directory_on_first_write(tmp_path, dataset):
    """Deferring creation must not simply break saving.

    ``save_model`` writes ``<model_artifacts_dir>/model_epoch_N.pth``. With eager creation
    gone, nothing else has made that directory, so ``save_model`` has to make it itself.
    The assertion before the call is the control: it shows the directory really was absent,
    so the file landing afterwards is evidence of lazy creation and not of a directory that
    was already sitting there.
    """
    artifacts = tmp_path / "runs" / "experiment1"

    model = Model(
        ModelConfig(vocab=VocabSpec.from_tokenizer(dataset.tokenizer), d_model=32, nhead=2, seed=5)
    )
    training_config = TrainingConfig(
        num_epochs=1, training_pathway="p2o", model_artifacts_dir=str(artifacts)
    )
    metrics_config = MetricsConfig(
        batch_metrics=False,
        training_metrics=False,
        validation_metrics=False,
        modes=[],
        filename=None,
    )
    pipeline = TrainingPipeline(
        model=model,
        dataset=dataset,
        training_config=training_config,
        metrics_logger=STDOutMetricsLogger(metrics_config),
    )

    assert not artifacts.exists(), "the artifacts directory was created before anything was saved"

    pipeline.save_model(epoch=0, run_name="run")

    assert (artifacts / "model_epoch_0.pth").is_file()
