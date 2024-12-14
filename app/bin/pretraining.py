import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
)
from src.domain.datamodels.dataset_config import DatasetConfig
from src.domain.datamodels.model_config import ModelConfig
from src.domain.datamodels.training_config import TrainingConfig
from src.domain.datamodels.wandb_config import WandbConfig
from src.utils.helper_funtions import handle_model_continuation

if __name__ == "__main__":

    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    model_config_handler = ModelConfigHandler(
        config_filepath="app/config/model_config.yaml"
    )
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()

    # wandb_config_handler = WandbConfigHandler(config_filepath="app/config/wandb_config.yaml")
    # wandb_config_handler.print_config()
    # wandb_config: WandbConfig = wandb_config_handler.get_config()

    training_config_handler = TrainingConfigHandler(
        config_filepath="app/config/training_config.yaml"
    )
    training_config_handler.print_config()
    training_config: TrainingConfig = training_config_handler.get_config()

    dataset_config_handler = DatasetConfigHandler(
        config_filepath="app/config/dataset_config.yaml"
    )
    dataset_config_handler.print_config()
    dataset_config: DatasetConfig = dataset_config_handler.get_config()

    for i in range(1, 3):
        if not os.path.exists(f"models/{i}"):
            os.mkdir(f"models/{i}")
        training_config.model_artifacts_dir = f"models/{i}"
        training_config.model_id = i
        dataset_config.dataset_filepath = f"data/pretraining/input_data_{i}.pkl"

        # wandb = WandbWrapper()
        # wandb.login()
        model_id, model_file_name = handle_model_continuation(training_config)
        train_model_handler = TrainModelHandler(
            model_config=model_config,
            dataset_config=dataset_config,
            training_config=training_config,
        )

        train_model_handler.initiate_model_training()
