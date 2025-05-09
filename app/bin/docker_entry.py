import os
import logging
from typing import Type, Tuple, Optional

from src.application.shared.base_config_handler import BaseConfigHandler
from src.infra.clients.gcp.gcs_client import GCSClient
from src.infra.data.storage_interface import StorageInterface

import sys
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

# Setup logging
logger = logging.getLogger(__name__)

def find_latest_checkpoint(bucket_name: str, task_index: int) -> Tuple[Optional[str], int]:
    """
    Find the latest checkpoint for a specific task in GCS.
    
    Args:
        bucket_name: The name of the GCS bucket
        task_index: The task index to look for checkpoints
        
    Returns:
        Tuple containing (checkpoint_path, latest_epoch) or (None, -1) if no checkpoint found
    """
    try:
        # Initialize the GCS client
        gcs_client = GCSClient(project="bridge-457501")
        
        # Define the prefix for model checkpoints
        prefix = f"pretraining/{task_index}/models/model_epoch_"
        logger.info(f"Looking for checkpoints with prefix: {prefix}")
        
        # Get underlying Google Cloud Storage client and bucket
        client = gcs_client.client
        bucket = client.bucket(bucket_name)
        
        # List all checkpoint blobs with the prefix
        blobs = list(bucket.list_blobs(prefix=prefix))
        logger.info(f"Found {len(blobs)} potential checkpoint files")
        
        if not blobs:
            logger.info("No existing checkpoints found")
            return None, -1
        
        # Extract epoch numbers from blob names
        latest_epoch = -1
        latest_blob = None
        
        for blob in blobs:
            # Extract epoch number from blob name
            file_name = blob.name.split('/')[-1]
            try:
                epoch = int(file_name.replace('model_epoch_', '').replace('.pth', ''))
                logger.info(f"Found checkpoint for epoch {epoch}")
                if epoch > latest_epoch:
                    latest_epoch = epoch
                    latest_blob = blob
            except ValueError:
                logger.warning(f"Couldn't parse epoch number from {file_name}")
                continue
        
        if latest_blob:
            # Create directory if it doesn't exist
            os.makedirs("tmp_checkpoints", exist_ok=True)
            
            # Download the latest checkpoint
            checkpoint_path = f"tmp_checkpoints/task_{task_index}_epoch_{latest_epoch}.pth"
            logger.info(f"Downloading latest checkpoint (epoch {latest_epoch}) to {checkpoint_path}")
            latest_blob.download_to_filename(checkpoint_path)
            return checkpoint_path, latest_epoch
        
        return None, -1
        
    except Exception as e:
        logger.exception(f"Error finding latest checkpoint: {e}")
        return None, -1

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
    
    # Initialize GCS client
    storage_interface = GCSClient(project="bridge-457501")
    
    # Load configs from GCS
    for key, handler_cls in handlers.items():
        config_filename = f"app/config/{key}.yaml"
        
        # Check if config exists in GCS and download if necessary
        bucket_name = os.environ["BUCKET_NAME"]
        if storage_interface.exists(bucket_name, f"pretraining/{key}.yaml"):
            logger.info(f"Downloading config from GCS: {key}.yaml")
            storage_interface.download_file(bucket_name, f"pretraining/{key}.yaml", config_filename)
        
        # Initialize handler and get config
        handler = handler_cls(config_filepath=config_filename)
        handler.print_config()
        configs[key] = handler.get_config()
    
    # Load dataset for current job task
    index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
    if not storage_interface.exists(os.environ["BUCKET_NAME"], f"pretraining/{index}/data_{index}.csv"):
        logger.error(f"Data file not found in GCS: pretraining/{index}/data_{index}.csv")
        raise FileNotFoundError(f"Data file not found in GCS: pretraining/{index}/data_{index}.csv")
    
    data_path = f"gs://{os.environ['BUCKET_NAME']}/pretraining/{index}/data_{index}.csv"
    logger.info(f"Setting dataset path to: {data_path}")
    configs["dataset_config"].dataset_filepath = data_path
    
    return configs

def main():
    # Set up logging first thing
    LoggingConfigHandler().setup_logging()
    logger.info("Starting training job")
    
    # Load all configuration
    configs = load_configs()
    
    # Get task index
    task_index = int(os.environ["CLOUD_RUN_TASK_INDEX"]) + 1
    logger.info(f"Running task {task_index}")
    
    try:
        # Find latest checkpoint for this task
        bucket_name = os.environ["BUCKET_NAME"]
        logger.info(f"Looking for checkpoints in bucket: {bucket_name} for task {task_index}")
        checkpoint_path, latest_epoch = find_latest_checkpoint(bucket_name, task_index)
        
        # Update training config with checkpoint path if found
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Resuming training from epoch {latest_epoch}")
            configs["training_config"].checkpoint_path = checkpoint_path
        else:
            logger.info("Starting training from scratch (no checkpoint found)")
        
        # Start training
        TrainModelHandler(**configs).initiate_model_training()
        
    except Exception as e:
        logger.exception(f"Error during training: {e}")
        raise
    finally:
        logger.info("Training job completed")


if __name__ == "__main__":
    main()