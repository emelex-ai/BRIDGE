import argparse  # NEW LIBRARY (pip install argparse)
from wandb_wrapper import WandbWrapper
from train import run_code
from dataset import ConnTextULDataset
import torch

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
    parser.add_argument("--test", action='store_true', default=False, help="Test mode: only run one epoch on a small subset of the data")
    parser.add_argument("--max_nb_steps", type=int, default=-1, help="Hardcode nb steps per epoch for fast testing")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Fraction of data in the training set")
    parser.add_argument("--sweep", type=str, required=True, default="", help="Run a wandb sweep from a file")
    
    args = parser.parse_args()
    
    num_epochs = args.num_epochs
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
    sweep = args.sweep
    assert d_model%nhead == 0, "d_model must be evenly divisible by nhead"

    if TEST:
        ds = ConnTextULDataset(test=True)
        d_model = 16
        nhead = 2
        num_layers = 2
        batch_size = 8
        learning_rate = 0.001
        num_epochs = 2
        max_nb_steps = -1
        CONTINUE = False
        seed = 1337
        train_test_split = 0.8
        # GE: I would prefer ref to torch not be in main.py
        torch.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)
        #torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        ds = ConnTextULDataset()

    #  Three parameters specific to W&B
    entity = "emelex"
    project = "ConnTextUL"
    is_wandb_enabled = True

    #  Parameters specific to the main code

    config = {
        # "starting_epoch": epoch_num,   # Add it back later once code is debugged
        "model_path": MODEL_PATH,
        "CONTINUE": CONTINUE,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "nhead": nhead,
        "learning_rate": learning_rate,
        "train_test_split": 0.8,   # <<< SET VIA ARGUMENT? 
        # "id": model_id,  # Add back later once code is debugged
        "common_num_layers": num_layers,
        # Set to -1 if all the steps should be executed
        "max_nb_steps": max_nb_steps,   # to speed up testing. Set to -1 to process full data. 
    }
    print("config: ", config)

    if args.sweep == True:
        wandb.set_params(
            config=config, is_sweep=True, is_wandb_on=is_wandb_enabled
        )  # GE: new function

        # make wandb wrapper accessible globally
        globals().update({"wandb": wandb})
        wandb.login()
        # Is it possible to update a sweep configuration? I'd like the default sweep
        # configuration to contain the parameters of config.
        # GE: suggestion: load different sweeps from files to keep track. 
        sweep_config = {
            "method": "grid",
            "name": "sweep_400ep_64d_m_128b",
            "metric": {
                'goal': 'minimize', 
                'name': 'time_per_epoch',
            },
            "parameters": {
                "batch_size": {"values": [128]},
                "d_model": {"values": [64]},
                "common_num_layers": {"values": [1, 4]},
            },
        }

        # Update sweep_config with new_params without overwriting existing parameters:
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, run_code)
    else:
        print("else")
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=is_wandb_enabled)

        globals().update({"wandb": wandb})
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
            print("wandb is disabled")
            run = wandb.init(config=config)
            # make sure I config is accessible with the dot notation
            run_code(ds)
        else:
            run_code(ds)


if __name__ == "__main__":
    main()
