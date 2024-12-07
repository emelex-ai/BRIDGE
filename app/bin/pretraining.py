import os
from app.bin.main import main
from src.application.handlers.dataset_config_handler import DatasetConfigHandler
from src.application.handlers.logging_config_handler import LoggingConfigHandler
from src.application.handlers.model_config_handler import ModelConfigHandler
from src.application.handlers.training_config_handler import TrainingConfigHandler
from src.application.handlers.wandb_config_handler import WandbConfigHandler
from src.domain.datamodels.dataset_config import DatasetConfig
from src.domain.datamodels.model_config import ModelConfig
from src.domain.datamodels.training_config import TrainingConfig
from src.domain.datamodels.wandb_config import WandbConfig

if __name__ == "__main__":
    
    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()


    # wandb_config_handler = WandbConfigHandler(config_filepath="app/config/wandb_config.yaml")
    # wandb_config_handler.print_config()
    # wandb_config: WandbConfig = wandb_config_handler.get_config()

    training_config_handler = TrainingConfigHandler(config_filepath="app/config/training_config.yaml")
    training_config_handler.print_config()
    training_config: TrainingConfig = training_config_handler.get_config()

    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config_handler.print_config()
    dataset_config: DatasetConfig = dataset_config_handler.get_config()


    for i in range(1,51):
        if not os.path.exists("src/domain/model/{i}"):
            os.mkdir(f"src/domain/model/{i}")
        training_config.model_artifacts_dir = f'src/domain/model/{i}'
        dataset_config.dataset_filepath = f'data/pretraining/input_data_{i}.pkl'
        main(model_config, dataset_config, training_config)