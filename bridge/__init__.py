"""BRIDGE: a computational model for naming printed words."""

from bridge.application.training import TrainingPipeline
from bridge.core.phonreps import PhonemeTable, load_phoneme_table
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    BridgeBatch,
    BridgeInput,
    BridgeTarget,
    DatasetConfig,
    GenerationOutput,
    ModelConfig,
    Target,
    Tokens,
    TrainingConfig,
    TrainingEvent,
    TrainingPhase,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.domain.model.model import PATHWAYS, Pathway
from bridge.domain.tokenizer import BridgeTokenizer, CharacterTokenizer, PhonemeTokenizer

__version__ = "0.1.0"

__all__ = [
    "PATHWAYS",
    "BridgeBatch",
    "BridgeDataset",
    "BridgeInput",
    "BridgeTarget",
    "BridgeTokenizer",
    "CharacterTokenizer",
    "DatasetConfig",
    "GenerationOutput",
    "Model",
    "ModelConfig",
    "Pathway",
    "PhonemeTable",
    "PhonemeTokenizer",
    "Target",
    "Tokens",
    "TrainingConfig",
    "TrainingEvent",
    "TrainingPhase",
    "TrainingPipeline",
    "VocabSpec",
    "load_phoneme_table",
]
