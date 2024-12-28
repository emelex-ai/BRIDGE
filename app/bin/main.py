import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.domain.datamodels import ModelConfig, DatasetConfig, WandbConfig, TrainingConfig
from src.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
)


def main(
    wandb_config: WandbConfig, model_config: ModelConfig, dataset_config: DatasetConfig, training_config: TrainingConfig
):
    train_model_handler = TrainModelHandler(
        wandb_config=wandb_config,
        model_config=model_config,
        dataset_config=dataset_config,
        training_config=training_config,
    )
    train_model_handler.initiate_model_training()


if __name__ == "__main__":

    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config = model_config_handler.get_config()

    wandb_config_handler = WandbConfigHandler(config_filepath="app/config/wandb_config.yaml")
    wandb_config_handler.print_config()
    wandb_config = wandb_config_handler.get_config()

    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config_handler.print_config()
    dataset_config = dataset_config_handler.get_config()

    training_config_handler = TrainingConfigHandler(config_filepath="app/config/training_config.yaml")
    training_config_handler.print_config()
    training_config = training_config_handler.get_config()

    main(wandb_config, model_config, dataset_config, training_config)
