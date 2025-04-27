import sys
import os

from src.infra.data.storage_interface import StorageInterface


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.application.handlers import (
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
    handlers = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
        "metrics_config": MetricsConfigHandler
    }
    configs = {}
    for key, handler_cls in handlers.items():
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        handler.print_config()
        configs[key] = handler.get_config()
    return configs


def main():
    configs = load_configs()
    storage_interface = StorageInterface()
    config_path = storage_interface.get_config()
    data_path = storage_interface.get_data()
    configs["training_config"]["data_path"] = data_path
    TrainModelHandler(**configs).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
