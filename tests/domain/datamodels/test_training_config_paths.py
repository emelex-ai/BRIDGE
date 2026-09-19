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
written. Creation moves to ``TrainingPipeline.save_checkpoint``, so the last test builds a real
pipeline and checks that saving still works once eager creation is gone.

``checkpoint_path`` is the control. It names a file that must already exist, so its
existence check has to survive; if that check disappeared along with the directory check,
a typo in a resume path would fail deep inside ``load_model`` instead of at construction.
"""

from pathlib import Path

import pytest

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.datamodels import (
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
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


def test_save_checkpoint_creates_the_artifacts_directory_on_first_write(tmp_path, words_dataset):
    """Deferring creation must not simply break saving.

    ``save_checkpoint`` resolves a relative path against ``model_artifacts_dir``. With eager
    creation gone, nothing else has made that directory, so the write has to make it.
    The assertion before the call is the control: it shows the directory really was absent,
    so the file landing afterwards is evidence of lazy creation and not of a directory that
    was already sitting there.
    """
    artifacts = tmp_path / "runs" / "experiment1"

    model = Model(
        ModelConfig(
            vocab=VocabSpec.from_tokenizer(words_dataset.tokenizer), d_model=32, nhead=2, seed=5
        )
    )
    training_config = TrainingConfig(training_pathway="p2o", model_artifacts_dir=str(artifacts))
    pipeline = TrainingPipeline(
        model=model,
        dataset=words_dataset,
        training_config=training_config,
    )

    assert not artifacts.exists(), "the artifacts directory was created before anything was saved"

    pipeline.save_checkpoint("model_epoch_0.pth", epoch=0)

    assert (artifacts / "model_epoch_0.pth").is_file()
