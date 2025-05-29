import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from bridge.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
    MetricsConfigHandler,
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
        handler = handler_cls(
            config_filepath=f"app/config/nuria_experiments/{key}.yaml"
        )
        handler.print_config()
        configs[key] = handler.get_config()
    return configs


def main(dataset_path: str):
    configs = load_configs()
    configs["dataset_config"].dataset_filepath = dataset_path
    configs["metrics_config"].filename = dataset_path.split("/")[-1]
    TrainModelHandler(**configs).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    dataset_paths = {
        # 1: "data/nuria_pretraining/p_p_bilingual_learner.csv",
        # 2: "data/nuria_pretraining/p_p_english_pred_learner.csv",
        3: "data/nuria_pretraining/p_p_spanish_pred_learner.csv",
    }
    for run, dataset_path in dataset_paths.items():
        print(f"Training with dataset: {dataset_path}")
        main(dataset_path=dataset_path)
