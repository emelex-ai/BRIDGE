import argparse  # NEW LIBRARY (pip install argparse)
from wandb_wrapper import WandbWrapper
from train import run_code
from dataset import ConnTextULDataset
import torch
import yaml

wandb = WandbWrapper()

def main():
    """ """
    parser = argparse.ArgumentParser(description='Train a ConnTextUL model')
    
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--continue_training", type=bool, default=False, help="Continue training from last checkpoint")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensionality of the internal model components \
                        including Embedding layer, transformer layers, \
                        and linear layers. Must be evenly divisible by nhead")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads for all attention modules. \
                        Must evenly divide d_model.")
    parser.add_argument("--wandb_disabled", type=bool, default=False, help="Disable wandb")
    parser.add_argument("--test", action='store_true', default=False, help="Test mode: only run one epoch on a small subset of the data")
    parser.add_argument("--max_nb_steps", type=int, default=-1, help="Hardcode nb steps per epoch for fast testing")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Fraction of data in the training set")

    parser.add_argument("--which_dataset", type=int, default=20, help="Choose the dataset to load")
    parser.add_argument("--sweep",type=str,  default="", help="Run a sweep from a configuration file")
    parser.add_argument("--d_embedding", type=int, default=1, help="Dimensionality of the final embedding layer.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for repeatibility.")

    args = parser.parse_args()
    
    wandb_disabled = args.wandb_disabled
    num_epochs = args.num_epochs
    d_embedding = args.d_embedding
    batch_size = args.batch_size
    num_layers = args.num_layers
    learning_rate = args.learning_rate
    CONTINUE = args.continue_training
    TEST = args.test
    nhead = args.nhead
    d_model = args.d_model
    MODEL_PATH = './models'
    max_nb_steps = args.max_nb_steps
    train_test_split = args.train_test_split
    which_dataset = args.which_dataset
    seed = args.seed
    assert d_model%nhead == 0, "d_model must be evenly divisible by nhead"

    #  Three parameters specific to W&B
    entity = "emelex"
    project = "ConnTextUL"
    is_wandb_enabled = not wandb_disabled

    if TEST:
        d_model = 16
        d_embedding = 2
        nhead = 2
        num_layers = 2
        batch_size = 8
        learning_rate = 0.001
        num_epochs = 2
        max_nb_steps = -1
        CONTINUE = False
        seed = 1337
        train_test_split = 1.0 # 0.8 (1.0 when using special test datasets)
        torch.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)
        #torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        is_wandb_enabled = False

    #  Parameters specific to the main code

    config = {
        # "starting_epoch": epoch_num,   # Add it back later once code is debugged
        "model_path": MODEL_PATH,
        "CONTINUE": CONTINUE,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "d_embedding": d_embedding,
        "nhead": nhead,
        "learning_rate": learning_rate,
        "train_test_split": train_test_split,   # <<< SET VIA ARGUMENT? 
        # "id": model_id,  # Add back later once code is debugged
        "common_num_layers": num_layers,
        # Set to -1 if all the steps should be executed
        "max_nb_steps": max_nb_steps,   # to speed up testing. Set to -1 to process full data. 
        "which_dataset": which_dataset, # select dataset from data/ folder
        "seed": seed, # select dataset from data/ folder
    }

    #print("config: ", config)
    globals().update({"wandb": wandb})

    if args.sweep != "":
        wandb.set_params(
            config=config, is_sweep=True, is_wandb_on=is_wandb_enabled
        )  # GE: new function

        wandb.login()

        with open(args.sweep, "r") as file:
            sweep_config = yaml.safe_load(file)

        #print("\n(BEFORE) sweep_config: ", sweep_config)

        # Update sweep_config with new_params without overwriting existing parameters:
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        #print("\n(AFTER) sweep_config: ", sweep_config)
        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)

        wandb.agent(sweep_id, run_code)
    else:
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=is_wandb_enabled)

        wandb.login()
        # 🐝 initialise a wandb run
        # I created a new project
        run = wandb.init(
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )
        # When 'disabled', the returned run.config is an empty dictionary {}
        if not is_wandb_enabled:
            run = wandb.init(config=config)
            run_code()
        else:
            run_code()


if __name__ == "__main__":
    main()
