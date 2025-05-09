import sys
import os
from typing import Type

from src.application.shared.base_config_handler import BaseConfigHandler
from src.infra.clients.gcp.gcs_client import GCSClient
from src.utils.helper_functions import find_latest_checkpoint


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.application.handlers import (
    ModelConfigHandler,
    DatasetConfigHandler,
    TrainingConfigHandler,
    WandbConfigHandler,
    LoggingConfigHandler,
    TrainModelHandler,
    MetricsConfigHandler,
)

storage_interface = GCSClient()


def load_configs():
    # Centralized config loading
    handlers: dict[str, Type[BaseConfigHandler]] = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
        "metrics_config": MetricsConfigHandler,
    }
    configs: dict[str, BaseConfigHandler] = {}
    # Load configs from GCS
    for key, handler_cls in handlers.items():
        if storage_interface.exists(
            os.environ["BUCKET_NAME"], f"pretraining/{key}.yaml"
        ):
            storage_interface.download_file(
                os.environ["BUCKET_NAME"],
                f"pretraining/{key}.yaml",
                f"app/config/{key}.yaml",
            )
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        handler.print_config()
        configs[key] = handler.get_config()
    # Load dataset for current job task
    index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
    if not storage_interface.exists(
        os.environ["BUCKET_NAME"], f"pretraining/{index}/data_{index}.csv"
    ):
        raise FileNotFoundError(
            f"Data file not found in GCS: pretraining/{index}/data_{index}.csv"
        )
    data_path = f"gs://{os.environ['BUCKET_NAME']}/pretraining/{index}/data_{index}.csv"
    configs["dataset_config"].dataset_filepath = data_path
    return configs


def main():
    configs = load_configs()

    # Find latest checkpoint for this task
    task_index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
    checkpoint_path, latest_epoch = find_latest_checkpoint(
        storage_interface, os.environ["BUCKET_NAME"], task_index
    )

    # Update training config with checkpoint path if found
    if checkpoint_path:
        logger.info(f"Resuming training from epoch {latest_epoch}")
        configs["training_config"].checkpoint_path = checkpoint_path

    TrainModelHandler(**configs).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
