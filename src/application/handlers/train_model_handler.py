from src.domain.datamodels import ModelConfig, DatasetConfig, TrainingConfig
from src.application.training import TrainingPipeline
from src.domain.dataset import BridgeDataset
from src.domain.model import Model
import logging
import time

logger = logging.getLogger(__name__)


class TrainModelHandler:

    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
    ):
        bridge_dataset = BridgeDataset(
            dataset_config=dataset_config, device=training_config.device
        )

        self.pipeline = TrainingPipeline(
            model=Model(
                model_config=model_config,
                dataset_config=dataset_config,
                device=training_config.device,
            ),
            training_config=training_config,
            dataset=bridge_dataset,
        )

    def initiate_model_training(self):
        self.pipeline.run_train_val_loop()
