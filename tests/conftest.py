"""Session-wide guards for the two pieces of global state BRIDGE tests can disturb.

Both exist because a test that leaks here fails a *different* test, which is the hardest
kind of failure to read.
"""

import os

# Popped at import, before `bridge` is imported below, and deliberately not in a fixture.
# `BRIDGE_DEVICE` selects the process device when `bridge.utils.device_manager` is first
# imported, which happens while pytest is collecting. A fixture runs long after that, so
# it would be reading an already-built singleton and could only claim to help. Pytest
# imports the rootdir conftest before any test module, so this is early enough.
# Measured: with BRIDGE_DEVICE=meta exported, popping here gives a clean suite where the
# fixture form gave 19 failed, 38 errors.
_BRIDGE_DEVICE = os.environ.pop("BRIDGE_DEVICE", None)

import pytest  # noqa: E402

from bridge.utils import device_manager  # noqa: E402


@pytest.fixture(autouse=True)
def restore_process_device():
    """Put ``device_manager`` back after any test that repoints it.

    Every BRIDGE object captures ``device_manager.device`` in its own ``__init__``, so a
    test that leaves the singleton on CUDA silently moves every model built after it. The
    device tests repoint it deliberately; this makes that safe rather than relying on each
    one to clean up.
    """
    before = device_manager.device
    yield
    device_manager._device = before


import pytest as _pytest  # noqa: E402,F811

from bridge.domain.data import BridgeDataset  # noqa: E402
from bridge.domain.datamodels import DatasetConfig, ModelConfig, TrainingConfig  # noqa: E402
from bridge.domain.model import Model  # noqa: E402
from bridge.domain.tokenizer import BridgeTokenizer  # noqa: E402

# The 7300-word corpus most pipeline tests train over.
WORDS_CSV = "tests/domain/model/data/data.csv"


@_pytest.fixture(scope="session")
def shared_tokenizer() -> BridgeTokenizer:
    """One tokenizer for the whole session.

    Thirteen test modules built their own. Construction is cheap after the first
    (0.96 ms against 392 ms cold, since the lexicon parse is memoised process-wide), so
    this is about having one definition rather than about speed.
    """
    return BridgeTokenizer()


@_pytest.fixture
def words_dataset(shared_tokenizer: BridgeTokenizer) -> BridgeDataset:
    """A ``BridgeDataset`` over ``WORDS_CSV``, sharing the session tokenizer."""
    return BridgeDataset(DatasetConfig(dataset_filepath=WORDS_CSV), tokenizer=shared_tokenizer)


@_pytest.fixture
def make_pipeline():
    """Build a ``TrainingPipeline`` over a dataset, overriding any ``TrainingConfig`` field.

    Six test modules had their own copy of this, so a new required config field broke
    eight call sites in six files. Model hyperparameters are small and seeded, matching
    what every copy used.
    """
    from bridge.application.training import TrainingPipeline

    def build(dataset: BridgeDataset, /, **overrides) -> TrainingPipeline:
        from bridge.domain.datamodels import VocabSpec

        model_kwargs = {
            key: overrides.pop(key)
            for key in ("d_model", "nhead", "seed", "d_embedding")
            if key in overrides
        }
        vocab = VocabSpec.from_tokenizer(dataset.tokenizer)
        return TrainingPipeline(
            model=Model(
                ModelConfig(vocab=vocab, **{"d_model": 16, "nhead": 2, "seed": 5, **model_kwargs})
            ),
            training_config=TrainingConfig(
                **{"num_epochs": 1, "training_pathway": "o2p", **overrides}
            ),
            dataset=dataset,
        )

    return build
