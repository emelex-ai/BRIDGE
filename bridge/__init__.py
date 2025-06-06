"""
BRIDGE: A computational model for naming printed words.

This package provides tools for training and using models that bridge
orthographic and phonological representations of words.
"""

from bridge.domain.model import Model
from bridge.domain.dataset import BridgeDataset, BridgeTokenizer
from bridge.domain.datamodels import (
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    WandbConfig,
    BridgeEncoding,
    GenerationOutput,
)
from bridge.application.training import TrainingPipeline
from bridge.application.handlers import TrainModelHandler

__version__ = "0.1.0"

__all__ = [
    "Model",
    "BridgeDataset",
    "BridgeTokenizer",
    "ModelConfig",
    "DatasetConfig",
    "TrainingConfig",
    "WandbConfig",
    "BridgeEncoding",
    "GenerationOutput",
    "TrainingPipeline",
    "TrainModelHandler",
]
