"""
BRIDGE: A computational model for naming printed words.

This package provides tools for training and using models that bridge
orthographic and phonological representations of words.
"""

from bridge.application.handlers import TrainModelHandler
from bridge.application.training import TrainingPipeline
from bridge.domain.datamodels import (
    BridgeEncoding,
    DatasetConfig,
    GenerationOutput,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)
from bridge.domain.dataset import BridgeDataset, BridgeTokenizer
from bridge.domain.model import Model

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
