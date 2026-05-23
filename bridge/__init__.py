"""
BRIDGE: A computational model for naming printed words.

This package provides tools for training and using models that bridge
orthographic and phonological representations of words.
"""

from bridge.application.training import TrainingPipeline
from bridge.domain.data import BridgeDataset
from bridge.domain.datamodels import (
    BridgeEncoding,
    DatasetConfig,
    EncodingComponent,
    GenerationOutput,
    MetricsConfig,
    ModelConfig,
    TrainingConfig,
    VocabSpec,
)
from bridge.domain.model import Model
from bridge.domain.tokenizer import BridgeTokenizer

__version__ = "0.1.0"

__all__ = [
    "BridgeDataset",
    "BridgeEncoding",
    "BridgeTokenizer",
    "DatasetConfig",
    "EncodingComponent",
    "GenerationOutput",
    "MetricsConfig",
    "Model",
    "ModelConfig",
    "TrainingConfig",
    "TrainingPipeline",
    "VocabSpec",
]
