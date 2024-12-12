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

sweep_config = {
    "program": "app/bin/pretraining.py",
    "method": "bayes",
    "name": "sweep_pretraining_0",
    "metric": {
        "goal": "minimize",
        "name": "loss"
    },
    "parameters": {
        "batch_size_train": {
            "values": [64]
        },
        "d_model": {
            "values": [64]
        },
        "d_embedding": {
            "values": [1]
        },
        "num_epochs": {
            "values": [50]
        },
        "learning_rate": {
            "values": [0.0001, 0.001]
        },
        "num_phon_enc_layers": {
            "values": [2]
        },
        "num_phon_dec_layers": {
            "values": [8]
        },
        "nhead": {
            "values": [4,8]
        },
        "weight_decay": {
            "values": [0.01,0.1,0.2,0.4]
        }
    } 
}


def pretrain():
    wandb.init()
    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    training_config_handler = TrainingConfigHandler(config_filepath="app/config/training_config.yaml")
    training_config: TrainingConfig = training_config_handler.get_config()
    training_config.use_wandb = True
    training_config.training_pathway = "p2p"
    training_config_dict = training_config.model_dump()
    for key in wandb.config.keys():
        if key in training_config_dict:
            training_config.__setattr__(key, wandb.config.__getattr__(key))
    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config: DatasetConfig = dataset_config_handler.get_config()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()
    TrainingPipeline.set_seed(model_config.seed)
    model_config_dict = model_config.model_dump()
    for key in wandb.config.keys():
        if key in model_config_dict:
            model_config.__setattr__(key, wandb.config.__getattr__(key))

    for i in range(48,49):
        if not os.path.exists(f"models/pretraining/{i}"):
            os.mkdir(f"models/pretraining/{i}")
        training_config.model_artifacts_dir = f'models/pretraining/{i}'
        dataset_config.dataset_filepath = f'data/pretraining/combined.pkl'
        main(model_config, dataset_config, training_config)
    wandb.finish()

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_config, project=os.environ.get("WAND_PROJECT", "BRIDGE"))
    #sweep_id='171xr0k5'
    wandb.agent(sweep_id, function=pretrain, project=os.environ.get("WAND_PROJECT", "BRIDGE"))
    