
from src.wandb_wrapper import WandbWrapper
from src.train import run_code
from src.dataset import ConnTextULDataset
import argparse 
import torch
import yaml
from attrdict import AttrDict
from typing import List, Tuple, Dict, Any, Union

wandb = WandbWrapper()

def read_args():
    parser = argparse.ArgumentParser(description='Train a ConnTextUL model')
    
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size_train", type=int, default=32, help="Train batch size")
    parser.add_argument("--batch_size_val", type=int, default=32, help="Validation batch size")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--continue_training", action='store_true', help="Continue training from last checkpoint \
                        (default: new run if argument absent)")
    parser.add_argument("--d_model", type=int, default=128, help="Dimensionality of the internal model components \
                        including Embedding layer, transformer layers, \
                        and linear layers. Must be evenly divisible by nhead")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads for all attention modules. \
                        Must evenly divided d_model.")
    # Be careful when using bool arguments. Must use action='store_true', which creates an option that defaults to True 
    parser.add_argument("--wandb", action='store_true', help="Enable wandb (default: disabled if argument absent)")
    parser.add_argument("--test", action='store_true', help="Test mode: only run one epoch on a small subset of the data \
                        (default: no test if argument absent)")
    parser.add_argument("--max_nb_steps", type=int, default=-1, help="Hardcode nb steps per epoch for fast testing")
    parser.add_argument("--train_test_split", type=float, default=0.8, help="Fraction of data in the training set")

    # which_dataset should only be used for testing
    parser.add_argument("--which_dataset", type=int, default=0, help="Choose the dataset to load")
    parser.add_argument("--sweep",type=str,  default="", help="Run a sweep from a configuration file")
    parser.add_argument("--d_embedding", type=int, default=1, help="Dimensionality of the final embedding layer.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for repeatibility.")
    parser.add_argument("--nb_samples", type=int, default=0, help="Number of total samples from dataset. All samples if <=0")
    parser.add_argument("--model_path", type=str, default="./models", help="Path to model checkpoint files.")

    args = parser.parse_args()
    if args.nb_samples <= 0:
        args.nb_samples = None

    if args.which_dataset <= 0:
        args.which_dataset = 'all'  # read all data

    if args.test == False:
        args.which_dataset = 'all'
    else:
        assert isinstance(args.which_dataset, int)

    return args
#----------------------------------------------------------------------
def hardcoded_args():
    # Overide progmra args with test dictionary
    # Do not include "test" in function names to avoid interference with pytest. 
    dct = AttrDict({})
    dct.d_model = 16
    dct.d_embedding = 2
    dct.nhead = 2
    dct.num_layers = 2
    dct.batch_size_train = 8
    dct.batch_size_val = 8
    dct.learning_rate = 0.001
    dct.num_epochs = 2
    dct.max_nb_steps = -1
    dct.continue_training = False
    dct.seed = 1337
    dct.nb_samples = 1000
    dct.train_test_split = 0.9
    dct.model_path = './models'
    dct.which_dataset = 100
    dct.test = True

    torch.manual_seed(dct.seed)  
    torch.cuda.manual_seed_all(dct.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dct

#----------------------------------------------------------------------
def main(args: Dict):
    """ """

    if args.wandb:
        wandb_enabled = True
    else:
        wandb_enabled = False

    config = AttrDict(args)
    assert config.d_model % config.nhead == 0, "d_model must be evenly divisible by nhead"

    #  Parameters specific to W&B
    entity = "emelex"
    project = "ConnTextUL"

    globals().update({"wandb": wandb})

    if args.sweep != "":
        wandb.set_params(
            config=config, is_sweep=True, is_wandb_on=wandb_enabled
        )  
        wandb.login()

        with open(args.sweep, "r") as file:
            sweep_config = yaml.safe_load(file)

        #"""
        # Update sweep_config with new_params without overwriting existing parameters:
        # Works. However, there are too many useless vertical lines in the hyperparameter parallel chart. 
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, run_code)
    else:
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=wandb_enabled)
        wandb.login()
        # 🐝 initialise a wandb run
        # I created a new project
        run = wandb.init(
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )
        # When 'disabled', the returned run.config is an empty dictionary {}
        if wandb_enabled:
            run = wandb.init(config=config)
            metrics = run_code()
        else:
            metrics = run_code()

    # Return data useful for testing. I would like the metrics at the last epoch
    return_dct = AttrDict({
        'metrics': metrics,
        'config': config,
    })

    return return_dct


#----------------------------------------------------------------------
if __name__ == "__main__":
    args_dct = AttrDict(vars(read_args()))
    test_dct = hardcoded_args()

    if args_dct.test:
        args_dct.update(test_dct)

    # I can now mock up the arguments for testing purposes
    return_dict = main(args_dct)
    metrics = return_dict.metrics
    print("\n==========================================")
    print("final metrics")
    for k, v in metrics.items():
        print("==> ", k, v)

#----------------------------------------------------------------------
