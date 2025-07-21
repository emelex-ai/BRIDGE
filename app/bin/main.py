import os
import sys

# from bridge.domain.model.utils import load_configs_dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.application.handlers import (
    DatasetConfigHandler,
    LoggingConfigHandler,
    MetricsConfigHandler,
    ModelConfigHandler,
    TrainingConfigHandler,
    TrainModelHandler,
    WandbConfigHandler,
)


def load_configs():
    # Centralized config loading
    handlers = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
        "metrics_config": MetricsConfigHandler,
    }
    configs = {}
    for key, handler_cls in handlers.items():
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        handler.print_config()
        configs[key] = handler.get_config()
    return configs


def main():
    configs = load_configs()
    # configs = load_configs_dict()
    TrainModelHandler(**configs).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
