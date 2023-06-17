from src.wandb_wrapper import WandbWrapper
from src.train import run_code
from src.dataset import ConnTextULDataset
import argparse
import torch
import yaml
import sys
from attrdict import AttrDict
from typing import List, Tuple, Dict, Any, Union

wandb = WandbWrapper()


def read_args():
    parser = argparse.ArgumentParser(description="Train a ConnTextUL model")

    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--batch_size_train", type=int, default=32, help="Train batch size"
    )
    parser.add_argument(
        "--batch_size_val", type=int, default=32, help="Validation batch size"
    )
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from last checkpoint \
                        (default: new run if argument absent)",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Dimensionality of the internal model components \
                        including Embedding layer, transformer layers, \
                        and linear layers. Must be evenly divisible by nhead",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads for all attention modules. \
                        Must evenly divided d_model.",
    )
    # Be careful when using bool arguments. Must use action='store_true', which creates an option that defaults to True
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable wandb (default: disabled if argument absent)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: only run one epoch on a small subset of the data \
                        (default: no test if argument absent)",
    )
    parser.add_argument(
        "--max_nb_steps",
        type=int,
        default=-1,
        help="Hardcode nb steps per epoch for fast testing",
    )
    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.8,
        help="Fraction of data in the training set",
    )

    # which_dataset should only be used for testing
    parser.add_argument(
        "--which_dataset", type=int, default=0, help="Choose the dataset to load"
    )
    parser.add_argument(
        "--sweep", type=str, default="", help="Run a sweep from a configuration file"
    )
    parser.add_argument(
        "--d_embedding",
        type=int,
        default=1,
        help="Dimensionality of the final embedding layer.",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed for repeatibility."
    )
    parser.add_argument(
        "--nb_samples",
        type=int,
        default=0,
        help="Number of total samples from dataset. All samples if <=0",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models",
        help="Path to model checkpoint files.",
    )
    parser.add_argument(
        "--pathway",
        type=str,
        default="op2op",
        help="Specify the particular pathway to use: o2p, p2o, op2op",
    )

    args = parser.parse_args()

    # Arguments on the command line
    # Works when test==False. When test==True, used_arguments should be tested against the
    #  hard-coded arguments
    
    used_arguments = {arg: value for arg, value in vars(args).items() if getattr(args, arg) != parser.get_default(arg)}

    if args.nb_samples <= 0:
        args.nb_samples = None

    if args.which_dataset <= 0:
        args.which_dataset = "all"  # read all data

    if args.test == False:
        args.which_dataset = "all"

    if args.sweep != "":
        args.wandb = True

    # Adding --wandb fixes error.
    # Perhaps add --wandb if sweep is set

    assert args.which_dataset == "all" or isinstance(args.which_dataset, int)

    assert args.pathway in [
        "o2p",
        "p2o",
        "op2op",
    ], "Invalid pathway argument: must be 'o2p', 'p2o', or 'op2op'"

    print("args: ", args)
    print("read_args: BEFORE RETURN")
    return args, used_arguments


# ----------------------------------------------------------------------
def hardcoded_args():
    # Parameter values used with test. Should override default arguments,
    # but should be overwritten by the arguments used when invoking the code.
    # Overide progrma args with test dictionary
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
    dct.model_path = "./models"
    dct.which_dataset = 100
    dct.test = True
    dct.pathway = 'op2op'
    dct.wandb = False
    dct.sweep = ""

    torch.manual_seed(dct.seed)
    torch.cuda.manual_seed_all(dct.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dct


# ----------------------------------------------------------------------
def main(args: Dict):
    """ """

    if args.wandb:
        wandb_enabled = True
    else:
        wandb_enabled = False

    config = AttrDict(args)
    assert (
        config.d_model % config.nhead == 0
    ), "d_model must be evenly divisible by nhead"

    #  Parameters specific to W&B
    entity = "emelex"
    project = "ConnTextUL"

    globals().update({"wandb": wandb})

    if args.sweep != "":
        wandb.set_params(config=config, is_sweep=True, is_wandb_on=wandb_enabled)
        wandb.login()

        with open(args.sweep, "r") as file:
            sweep_config = yaml.safe_load(file)

        # Update sweep_config with new_params without overwriting existing parameters:
        # Works. However, there are too many useless vertical lines in the hyperparameter parallel chart.
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        run_code1 = run_code(args)
        wandb.agent(sweep_id, run_code1)
        run = wandb.init()
        print("run.config: ", run.config)
    else:
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=wandb_enabled)
        wandb.login()
        # 🐝 initialise a wandb run
        # I created a new project
        run = wandb.init(
            name=wandb_name, # Name of run that reflects model and run number
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )
        # When 'disabled', the returned run.config is an empty dictionary {}
        if wandb_enabled:
            # Add name argument which should be the same as the model file
            run = wandb.init(config=config, name=wandb_name)
            metrics = run_code(args)
        else:
            metrics = run_code(args)

    # Return data useful for testing. I would like the metrics at the last epoch
    # return_dct = AttrDict({
    #'config': run.config
    # })

    # return return_dct
    return 0

#----------------------------------------------------------------------
def handle_arguments():
    if '--test' in sys.argv:
        test_mode = True
    else:
        test_mode = False

    args, used_args_dct = read_args()
    args_dct = AttrDict(vars(args))
    test_dct = hardcoded_args()

    # Compute used args by comparing arguments against sys.argv
    # Assumes that all arguments are prepended with '--'
    used_test_keys = [k for k in vars(args).keys() if '--'+k in sys.argv]
    used_test_dct = {k: args_dct[k] for k in used_test_keys}

    if test_mode == True:
        # overwrite arg dict with hardcoded test values
        args_dct.update(test_dct)
        # overwrite arg dict with actual arguments used
        args_dct.update(used_test_dct)

    return args_dct


# ----------------------------------------------------------------------
if __name__ == "__main__":

    args_dct = handle_arguments()
    for k, v in args_dct.items():
        print(f"{k} ==> {v}")

    status = main(args_dct)
    if status == 0:
        print("Return from main: No errors")

