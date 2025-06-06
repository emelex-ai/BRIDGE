import sys
import os

from bridge.domain.datamodels.dataset_config import DatasetConfig
from bridge.domain.datamodels.metrics_config import MetricsConfig
from bridge.domain.datamodels.model_config import ModelConfig
from bridge.domain.datamodels.training_config import TrainingConfig
from bridge.domain.datamodels.wandb_config import WandbConfig


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
    MetricsConfigHandler
)


def load_configs():
    # Centralized config loading
    wandb_config: WandbConfig = WandbConfigHandler("app/config/wandb_config.yaml").get_config()
    model_config: ModelConfig = ModelConfigHandler("app/config/model_config.yaml").get_config()
    dataset_config: DatasetConfig = DatasetConfigHandler("app/config/dataset_config.yaml").get_config()
    training_config: TrainingConfig = TrainingConfigHandler("app/config/training_config.yaml").get_config()
    metrics_config: MetricsConfig = MetricsConfigHandler("app/config/metrics_config.yaml").get_config()
    return wandb_config, model_config, dataset_config, training_config, metrics_config



def main():
    wandb_config, model_config, dataset_config, training_config, metrics_config = load_configs()
    training_config.training_pathway = "o2p"
    for i in range(22): # Training datasets
        dataset_config.dataset_filepath = "data/training/{i}"
        for j in range(50): # Pretrained models
            training_config.checkpoint_path = f"models/pretraining/{j}/model_epoch_950.pth"
            
    TrainModelHandler(wandb_config=wandb_config,
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
        metrics_config=metrics_config
    ).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
