import sys
import os
from typing import Type

from src.application.shared.base_config_handler import BaseConfigHandler
from src.infra.clients.gcp.gcs_client import GCSClient
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

storage_interface = GCSClient()

def load_configs():
    # Centralized config loading
    handlers: dict[str, Type[BaseConfigHandler]] = {
        "wandb_config": WandbConfigHandler,
        "model_config": ModelConfigHandler,
        "dataset_config": DatasetConfigHandler,
        "training_config": TrainingConfigHandler,
        "metrics_config": MetricsConfigHandler
    }
    configs: dict[str, BaseConfigHandler] = {}
    # Load configs from GCS
    for key, handler_cls in handlers.items():
        if storage_interface.exists(os.environ["BUCKET_NAME"], f"pretraining/{key}.yaml"):
            storage_interface.download_file(os.environ["BUCKET_NAME"], f"pretraining/{key}.yaml", f"app/config/{key}.yaml")
        handler = handler_cls(config_filepath=f"app/config/{key}.yaml")
        handler.print_config()
        configs[key] = handler.get_config()
    # Load dataset for current job task
    index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
    if not storage_interface.exists(os.environ["BUCKET_NAME"], f"pretraining/{index}/data_{index}.csv"):
        raise FileNotFoundError(f"Data file not found in GCS: pretraining/{index}/data_{index}.csv")
    data_path = f"gs://{os.environ['BUCKET_NAME']}/pretraining/{index}/data_{index}.csv"
    configs["dataset_config"].dataset_filepath = data_path
    return configs


def main():
    configs = load_configs()
    
    TrainModelHandler(**configs).initiate_model_training()


if __name__ == "__main__":
    LoggingConfigHandler().setup_logging()
    main()
