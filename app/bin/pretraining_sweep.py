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
            "values": [32,64]
        },
        "batch_size_val": {
            "values": [32,64]
        },
        "d_model": {
            "values": [128]
        },
        "d_embedding": {
            "values": [1]
        },
        "num_epochs": {
            "values": [100]
        },
        "learning_rate": {
            "values": [0.0001, 0.001]
        },
        "num_phon_enc_layers": {
            "values": [8,16,32]
        },
        "num_phon_dec_layers": {
            "values": [2,4,8,16,32]
        },
        "nhead": {
            "values": [8,16,32]
        },
        "weight_decay": {
            "values": [0.01,0.1]
        }
    } 
}


def pretrain():
    '''
    Uses wandb to sweep through optimal hyperparameters for pretraining.
    Uses the first dataset for pretraining.
    '''
    wandb.init()
    logging_config_handler = LoggingConfigHandler()
    logging_config_handler.setup_logging()

    training_config_handler = TrainingConfigHandler(config_filepath="app/config/training_config.yaml")
    training_config: TrainingConfig = training_config_handler.get_config()
    training_config.use_wandb = True
    training_config.training_pathway = "p2p"
    # Replace the training parameters with any overridden values from the wandb run 
    training_config_dict = training_config.model_dump()
    for key in wandb.config.keys():
        if key in training_config_dict:
            training_config.__setattr__(key, wandb.config.__getattr__(key))
    dataset_config_handler = DatasetConfigHandler(config_filepath="app/config/dataset_config.yaml")
    dataset_config: DatasetConfig = dataset_config_handler.get_config()

    model_config_handler = ModelConfigHandler(config_filepath="app/config/model_config.yaml")
    model_config_handler.print_config()
    model_config: ModelConfig = model_config_handler.get_config()
    if model_config.seed is not None:
        TrainingPipeline.set_seed(model_config.seed)
    model_config_dict = model_config.model_dump()
    for key in wandb.config.keys():
        if key in model_config_dict:
            model_config.__setattr__(key, wandb.config.__getattr__(key))

    if not os.path.exists(f"models/pretraining/1"):
        os.mkdir(f"models/pretraining/1")
    training_config.model_artifacts_dir = f'models/pretraining/1'
    dataset_config.dataset_filepath = f'data/pretraining/input_data_1.pkl'
    main(model_config, dataset_config, training_config)
    wandb.finish()

if __name__ == "__main__":
    wandb.login()
    sweep_id = wandb.sweep(sweep=sweep_config, project=os.environ.get("WAND_PROJECT", "BRIDGE"))
    wandb.agent(sweep_id, function=pretrain, project=os.environ.get("WAND_PROJECT", "BRIDGE"))
    