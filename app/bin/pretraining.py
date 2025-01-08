import os

import wandb
from app.bin.main import main
from src.application.handlers.dataset_config_handler import DatasetConfigHandler
from src.application.handlers.logging_config_handler import LoggingConfigHandler
from src.application.handlers.model_config_handler import ModelConfigHandler
from src.application.handlers.training_config_handler import TrainingConfigHandler
from src.application.handlers.wandb_config_handler import WandbConfigHandler
from src.application.training.training_pipeline import TrainingPipeline
from src.domain.datamodels.dataset_config import DatasetConfig
from src.domain.datamodels.model_config import ModelConfig
from src.domain.datamodels.training_config import TrainingConfig
from src.domain.datamodels.wandb_config import WandbConfig

def pretrain():
    '''
    Runs p2p pretraining on each of the 50 pretraining datasets and saves trained models in separate directories
    '''
    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    training_config_handler = TrainingConfigHandler(config_filepath="app/config/training_config.yaml")
    training_config: TrainingConfig = training_config_handler.get_config()
    training_config.training_pathway = "p2p"
    training_config.device = "cpu"
    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config: DatasetConfig = dataset_config_handler.get_config()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()

    if model_config.seed:
        TrainingPipeline.set_seed(model_config.seed)
     
    wandb.login()
    for i in range(13,50):
        wandb.init(project="BRIDGE"
        , name=f"pretraining_{i}", config = {"model_id": f"pretraining_{i}"})
        if not os.path.exists(f"models/pretraining/{i}"):
            os.mkdir(f"models/pretraining/{i}")
        training_config.model_artifacts_dir = f'models/pretraining/{i}'
        dataset_config.dataset_filepath = f'data/pretraining/input_data_{i}.pkl'
        main(model_config, dataset_config, training_config)

if __name__ == "__main__":
    pretrain()
    