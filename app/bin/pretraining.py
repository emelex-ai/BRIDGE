import os
import sys

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
        #handler.print_config()
        configs[key] = handler.get_config()
    return configs


def main():
    configs = load_configs()
    for dataset_num in range(19, 30):
        dataset_config = configs["dataset_config"]
        parts = dataset_config.dataset_filepath.split("/")
        parts[-1] = f"input_data_{dataset_num}.pkl" 
        dataset_config.dataset_filepath = "/".join(parts)
        configs["dataset_config"] = dataset_config
        TrainModelHandler(**configs).initiate_model_pretraining()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
