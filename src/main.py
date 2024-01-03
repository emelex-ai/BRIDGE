"""
This module contains the main function for training a ConnTextUL model. It reads 
command-line arguments, sets up the training environment, and runs the training 
loop. The module also defines a function for setting up hardcoded arguments for 
testing purposes.

Functions:
- read_args(): Parses command-line arguments and returns a dictionary of arguments.
- hardcoded_args(): Returns a dictionary of hardcoded arguments for testing purposes.
- main(args: Dict): Runs the training loop with the given arguments.
"""
from src.wandb_wrapper import WandbWrapper
from src.train import run_code, run_code_sweep
import src.train_impl as train_impl
from src.train_impl import get_new_model_id, get_user, get_model_file_name, get_latest_run, extract_model_id_epoch
from src.dataset import ConnTextULDataset
import argparse
import torch
import yaml
import sys, os
from addict import Dict as AttrDict
from typing import List, Tuple, Dict, Any, Union
from pprint import pprint

# used to get the user name in a portable manner

wandb = WandbWrapper()


def read_args():
    parser = argparse.ArgumentParser(description="Train a ConnTextUL model")

    parser.add_argument(
        "--device", type=str, default='cpu', help="cpu or gpu device"
    )
    parser.add_argument(
        "--project", type=str, required=True, help="Project name (no default)"
    )
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
        "--model_chkpt",
        default="",
        help="Continue training from the checkpoint model_chkpt \n \
                (assumes --continue_training is present)\n \
                (default: continue from latest run if --model_id is absent)",
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

    parser.add_argument(
        "--save_every",
        type=int,
        default=1, 
        help="Save data every 'save_every' number of epochs. Default: 1", 
    )

    parser.add_argument(
        "--input_data",
        type=str,
        default="data.csv",
        help="Name of the input training file present in the /data folder. Can be a path with root folder as the /data folder."
    )

    args = parser.parse_args()

    if args.nb_samples <= 0:
        args.nb_samples = None

    if args.which_dataset <= 0:
        args.which_dataset = "all"  # read all data

    if args.test == False:
        args.which_dataset = "all"

    if args.sweep != "":
        args.wandb = True

    assert args.which_dataset == "all" or isinstance(args.which_dataset, int)

    assert args.pathway in [
        "o2p",
        "p2o",
        "op2op",
    ], "Invalid pathway argument: must be 'o2p', 'p2o', or 'op2op'"

    return args


# ----------------------------------------------------------------------
def hardcoded_args():
    # Parameter values used with test. Should override default arguments,
    # but should be overwritten by the arguments used when invoking the code.
    # Overide program args with test dictionary
    # Do not include "test" in function names to avoid interference with pytest.
    dct = AttrDict({})
    dct.device = 'cpu'
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
    dct.model_chkpt = ""
    dct.seed = 1337
    dct.nb_samples = 1000
    dct.train_test_split = 0.9
    dct.model_path = "./models"
    dct.which_dataset = 100
    dct.test = True
    dct.pathway = "op2op"
    #dct.epochs_completed = 0  # should probably not be in dict. config attrib should not change
    # wandb should always be set through the command line. False by default.
    dct.sweep = ""
    dct.save_every = 2
    dct.input_data = "data.csv"

    torch.manual_seed(dct.seed)
    torch.cuda.manual_seed_all(dct.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return dct

# ----------------------------------------------------------------------
def setup_new_run(config):
    
    model_id = get_new_model_id()
    model_file_name = get_model_file_name(model_id, epoch_num=0)
    config.model_id = model_id
    #print(f"{config.continue_training=}, {model_file_name=}")
    # Probably should save the new file at initial time. NOT DONE.
    epochs_completed = 0  # new run (not a continuation run)
    config.chkpt_file_exists = False
    return model_id, model_file_name, epochs_completed

# ----------------------------------------------------------------------
def handle_model_continuation(config):
    ##############################
    #  Handle potential model continuation
    ##############################

    if config.continue_training == False:
        # New simulation with new model_id
        dct = hardcoded_args()
        model_id, model_file_name, epochs_completed = setup_new_run(config)
    else:
        # Continue existing simulation
        if config.model_chkpt:
            # Continue from this model_chkpt file
            # Check whether the file exists
            # If the file does not exist
            full_path = os.path.join(config.model_path, config.model_chkpt)
            if os.path.exists(full_path):
                # Desired checkpointed file exists
                model_file_name = config.model_chkpt  
                model_id, epochs_completed = extract_model_id_epoch(model_file_name)
                config.chkpt_file_exists = True
            else:
                print("Exit program")
                quit()
        else:
            # Continue from the last model id of the current user
            user = get_user()
            try:
                latest_file = get_latest_run(config.model_path, user)  # latest_file exists
                model_id, epochs_completed = extract_model_id_epoch(latest_file)
                model_file_name = latest_file
                config.chkpt_file_exists = True
            except:
                # If there is no latest file from the user in the proper format, 
                print("Expected a continuation run, but there is no previous run")
                quit()

    # model_file_name: file name to start from (either end of previous run or an initial run)
    # The end of previous run will be stored, the initial run is not yet stored
    # epochs_completed: if zero, it is a new run
    # model_id: username + date (yyyy-mm-dd) + time (14h17m23232ms)
    return model_id, epochs_completed, model_file_name
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

    model_id, epochs_completed, model_file_name = handle_model_continuation(config)

    # Not clear these are needed, but they can't hurt. They are unchanged during the run
    config.model_id = model_id
    config.epochs_completed = epochs_completed 
    config.model_file_name = model_file_name

    #  Parameters specific to W&B
    entity = "emelex"
    project = args.project  # "GE_ConnTextUL"

    globals().update({"wandb": wandb})
    print("==> main, after globals")


    if args.sweep != "":

        #################################
        # Perform parameter sweep 
        #################################

        print("==> Perform parameter sweep")
        wandb.set_params(config=config, is_sweep=True, is_wandb_on=True)
        wandb.login()

        #wandb.read_wandb_history(entity, project)  # Add for sweeps and tests as well
        #raise "error"

        with open(args.sweep, "r") as file:
            sweep_config = yaml.safe_load(file)

        # parameters in sweep with more than a single value
        keys = []  # List of keys with more than one value in sweep file
        for k, v in sweep_config["parameters"].items():
            try:
                if len(v["values"]) > 1:
                    keys.append(k)
            except:
                # no "values" key
                pass
        print("Sweep parameter keys with more than one value: ", keys)

        # Update sweep_config with new_params without overwriting existing parameters:
        # There are now too many useless vertical lines in the hyperparameter parallel chart.
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, function=lambda: run_code_sweep(args))
    else:

        #################################
        #  do NOT perform parameter sweep.
        #  Wandb might be turned on or off, but write code as if wandb is active.
        #################################

        print("main: do NOT perform parameter sweep")
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=wandb_enabled)
        wandb.login()

        """ (2023-10-01)
        print(f"==> {args_dct.model_path=}")
        model_id, epoch_num = train_impl.get_starting_model_epoch(
            args_dct.model_path, continue_training=args_dct.continue_training
        )
        """

        epochs_completed = config.epochs_completed

        #wandb_name = train_impl.get_model_file_name(model_id, epoch_num)
        wandb_name = config.model_id
        print("wandb_name, filename: ", wandb_name)

        run = wandb.init(
            name=wandb_name,  # Let wandb choose the name
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )

        #run_id = run.id
        #name_id = run.name
        #run_id = run.id
        #print("main: type run.id: ", run.id)
        #print("main: type run.id: ", run.name)

        #print("version else")
        #wandb.read_wandb_history(entity, project, run)  # Add for sweeps and tests as well
        #raise "error_else"

        #metrics = run_code(run, epoch_num, model_id)
        metrics = run_code(run, epochs_completed, model_id)

    return 0


# ----------------------------------------------------------------------
def handle_arguments():
    """
    If in test =mode (--test): 
        Use the hardcoded variables in hardcoded_args()
        Override these arguments with command line arguments

    If not in test mode: 
        Use command line arguments only
        
    """
    if "--test" in sys.argv:
        test_mode = True
    else:
        test_mode = False

    args = read_args()
    args_dct = AttrDict(vars(args))
    # will be overwritten if loading a saved file
    #args_dct.epochs_completed = 0  # 2023-09-18
    test_dct = hardcoded_args()
    print("==========================")
    print(f"Hardcoded, test_dct=")
    pprint(test_dct)
    print("==========================")

    # When not in test mode, arguments overwrite the default values
    # When in test mode, arguments overwrite the values in hardcoded_args

    # Compute used args by comparing arguments against sys.argv
    # Assumes that all arguments are prepended with '--'
    used_test_keys = [k for k in vars(args).keys() if "--" + k in sys.argv]
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
