"""
BRIDGE: A computational model for naming printed words.

This package provides tools for training and using models that bridge
orthographic and phonological representations of words.
"""

from bridge.application.training import TrainingPipeline
from bridge.core.phonreps import PhonemeTable, load_phoneme_table
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    BridgeEncoding,
    DatasetConfig,
    EncodingComponent,
    GenerationOutput,
    ModelConfig,
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
    "BridgeDataset",
    "BridgeEncoding",
    "BridgeTokenizer",
    "CharacterTokenizer",
    "DatasetConfig",
    "EncodingComponent",
    "GenerationOutput",
    "Model",
    "ModelConfig",
    "Pathway",
    "PhonemeTable",
    "PhonemeTokenizer",
    "TrainingConfig",
    "TrainingEvent",
    "TrainingPhase",
    "TrainingPipeline",
    "VocabSpec",
    "load_phoneme_table",
]
