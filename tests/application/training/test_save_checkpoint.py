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
import pathlib

import pytest
import torch
from pydantic import ValidationError

from bridge.application.training.training_pipeline import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    DatasetConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model

DATA_CSV = "tests/domain/model/data/data.csv"


@pytest.fixture(scope="module")
def dataset():
    """Parsing the 7300-word csv is the slow part, so it happens once."""
    return BridgeDataset(dataset_config=DatasetConfig(dataset_filepath=DATA_CSV))


def make_pipeline(dataset, artifacts_dir, **overrides):
    vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
    return TrainingPipeline(
        model=Model(ModelConfig(vocab=vocab, d_model=16, nhead=2, seed=5)),
        training_config=TrainingConfig(
            training_pathway="o2p",
            model_artifacts_dir=str(artifacts_dir),
            **overrides,
        ),
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
    # Recorded only when the caller names the dataset. The pipeline no longer holds one,
    # so it cannot stamp a bundle with data that did not train the weights.
    assert "dataset_config" not in loaded

    with_data = pipeline.save_checkpoint(tmp_path / "with_data.pth", epoch=7, dataset=dataset)
    reloaded = torch.load(with_data, weights_only=False)
    assert reloaded["dataset_config"].dataset_filepath == dataset.dataset_config.dataset_filepath


def test_nothing_writes_a_checkpoint_except_save_checkpoint(words_dataset, make_pipeline, tmp_path):
    """The library owns no save policy, which is now a structural fact rather than a
    property of one loop: there is no loop left to check.

    Oracle: a search with a stated space, parsed rather than grepped, and the space is
    the whole module rather than the bodies of plain functions. The first version of this
    walked only `ast.FunctionDef` for a literal `torch.save` attribute, so a call at module
    scope, a call inside an `async def`, and `from torch import save` all passed it while
    claiming to have searched for "every torch.save call". Validated by planting each of
    those three spellings plus the ordinary one and confirming all four are caught.
    """
    import ast

    from tests.conftest import batch_slices

    def saving_functions(tree: ast.AST) -> list[str]:
        """Names of the enclosing defs, or '<module>', for every call that saves a tensor."""
        aliased = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "torch"
            for alias in node.names
            if alias.name == "save"
        }
        enclosing: dict[ast.AST, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    enclosing.setdefault(child, node.name)
        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            saves = (
                isinstance(func, ast.Attribute)
                and func.attr == "save"
                and isinstance(func.value, ast.Name)
                and func.value.id == "torch"
            ) or (isinstance(func, ast.Name) and func.id in aliased)
            if saves:
                found.append(enclosing.get(node, "<module>"))
        return found

    callers = [
        f"{path}:{name}"
        for path in sorted(pathlib.Path("bridge").rglob("*.py"))
        for name in saving_functions(ast.parse(path.read_text()))
    ]
    assert callers == ["bridge/application/training/training_pipeline.py:save_checkpoint"], callers

    pipeline = make_pipeline(words_dataset, model_artifacts_dir=str(tmp_path))
    list(pipeline.train_steps(words_dataset, batch_slices(words_dataset)[:2], epoch=0))

    assert list(tmp_path.rglob("*.pth")) == [], "a step wrote a checkpoint nobody asked for"


def test_save_every_is_gone(tmp_path):
    """A leftover cadence field would quietly reintroduce library-owned save policy.

    Three checks, because one is not enough. The field is off the schema; a config handed
    one anyway is rejected rather than quietly ignored, so a stale `save_every=2` in
    someone's experiment config is a loud failure instead of a run that silently saves
    nothing; and no code reads it. The last is the one that matters, and it is a search
    with a stated space: the two modules that ever referred to it.
    """
    assert "save_every" not in TrainingConfig.model_fields

    with pytest.raises(ValidationError, match="save_every"):
        TrainingConfig(model_artifacts_dir=str(tmp_path), save_every=2)

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
