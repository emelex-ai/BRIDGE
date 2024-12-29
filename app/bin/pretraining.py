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


def load_configs():
    # Centralized config loading
    handlers = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
    }
    configs = {}
    for key, handler_cls in handlers.items():
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        handler.print_config()
        configs[key] = handler.get_config()
    return configs


def main():
    configs = load_configs()
    TrainModelHandler(**configs).initiate_model_pretraining()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
