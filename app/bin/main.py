import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.configuration.model import ModelConfig, ModelConfigHandler
from src.utils.configuration.logging import LoggingConfigHandler
from src.utils.configuration.dataset import DatasetConfig, DatasetConfigHandler
from src.utils.configuration.wandb import WandbConfig, WandbConfigHandler
from src.utils.helper_funtions import handle_model_continuation
from src.infra.clients.wandb import WandbWrapper



def main(configs: dict):
    wandb = WandbWrapper()
    wandb.login()
    model_id, model_file_name = handle_model_continuation(configs.get("model_config"))


if __name__ == "__main__":

    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()

    wandb_config_handler = WandbConfigHandler(config_filepath="app/config/wandb_config.yaml")
    wandb_config_handler.print_config()
    wandb_config: WandbConfig = wandb_config_handler.get_config()

    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config_handler.print_config()
    dataset_config: DatasetConfig = dataset_config_handler.get_config()

    configs: dict = {
        "model_config": model_config,
        "wandb_config": wandb_config,
        "dataset_config": dataset_config,
    }
    main(configs)
