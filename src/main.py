"""
This module contains the main function for training a ConnTextUL model. It reads
in a configuration file, sets up the training environment, and runs the training 
loop. 
"""

from src.wandb_wrapper import WandbWrapper
from src.train import run_code, run_code_sweep
from src.train_impl import (
    get_new_model_id,
    get_model_file_name,
)
import argparse
import yaml
import sys
from pathlib import Path
import os
from addict import Dict as AttrDict
from typing import List, Tuple, Dict, Any, Union

# used to get the user name in a portable manner

wandb = WandbWrapper()


def read_args():
    """Parse command-line arguments and return a dictionary of arguments.

    Program expexts one of the options:
        1) Nothing passed in, in which case the program uses hardcoded arguments
        2) A config file passed in, in which case the program uses the arguments in the file
    """

    parser = argparse.ArgumentParser(description="Train a ConnTextUL model")

    # Add arguments to the parser
    parser.add_argument("--config", type=str, help="Path to config file")

    args = AttrDict(vars(parser.parse_args()))

    return args


# ----------------------------------------------------------------------
def setup_new_run(config):
    model_id = get_new_model_id()
    config.model_id = model_id
    model_file_name = get_model_file_name(model_id, 0)
    return model_id, model_file_name


# ----------------------------------------------------------------------
def handle_model_continuation(config):
    ##############################
    #  This function will be used to handle the scenario where we wish
    #  to continue training an existing model. This is not yet implemented.
    ##############################

    # Add logic here to handle continuation of a model

    # For now, setup new simulation with new model_id
    model_id, model_file_name = setup_new_run(config)
    print(f"New run: {model_file_name=}")

    return model_id, model_file_name


# ----------------------------------------------------------------------
def main(config: AttrDict):
    """ """

    if config.wandb:
        wandb_enabled = True
    else:
        wandb_enabled = False

    model_id, model_file_name = handle_model_continuation(config)

    # Not clear these are needed, but they can't hurt. They are unchanged during the run
    config.model_id = model_id
    config.model_file_name = model_file_name

    #  Parameters specific to W&B
    entity = "nathan-crock"
    project = config.project

    globals().update({"wandb": wandb})
    print("==> main, after globals")

    if config.sweep_filename != "":
        assert (
            wandb_enabled
        ), "For a parameter sweep, Wandb must be enabled in the config file"

        #################################
        # Perform parameter sweep
        #################################

        print("==> Perform parameter sweep")
        wandb.set_params(config=config, is_sweep=True, is_wandb_on=True)
        wandb.login()

        # Attempt to parse the sweep file using the read_yaml function (basic error checking)
        sweep_config = read_yaml(config.sweep_filename)

        # Update sweep_config with new_params without overwriting existing parameters:
        # There are now too many useless vertical lines in the hyperparameter parallel chart.
        for param, value in config.items():
            if param not in sweep_config["parameters"]:
                sweep_config["parameters"][param] = {"values": [value]}

        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, function=lambda: run_code_sweep(config))
    else:

        #################################
        #  do NOT perform parameter sweep.
        #  Wandb might be turned on or off, but write code as if wandb is active.
        #################################

        print("main: do NOT perform parameter sweep")
        wandb.set_params(config=config, is_sweep=False, is_wandb_on=wandb_enabled)
        wandb.login()

        wandb_name = config.model_id
        print("wandb_name, filename: ", wandb_name)

        run = wandb.init(
            name=wandb_name,  # Let wandb choose the name
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )

        # Nothing done with returned metrics yet... If not needed, consider not returning
        metrics = run_code(run, model_id)

    return 0


def load_config(config_filepath: str = None):
    # Default test values. If there is no config file passed, code uses these values. Helpful
    # for quick testing when making minor code changes
    default_values = AttrDict(
        {
            "device": "cpu",
            "project": "TestProject",
            "num_epochs": 2,
            "batch_size_train": 32,
            "batch_size_val": 32,
            "num_phon_enc_layers": 2,
            "num_orth_enc_layers": 2,
            "num_mixing_enc_layers": 2,
            "num_phon_dec_layers": 2,
            "num_orth_dec_layers": 2,
            "learning_rate": 0.001,
            "d_model": 64,
            "nhead": 2,
            "wandb": False,
            "train_test_split": 0.8,
            "sweep_filename": "",
            "d_embedding": 1,
            "seed": 1337,
            "model_path": "./models",
            "pathway": "o2p",
            "save_every": 1,
            "dataset_filename": "data.csv",
            "max_nb_steps": 10,
            "test_filenames": ["test1.csv", "test2.csv"],
        }
    )

    # Grab the project root for relative path conversion
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if not config_filepath:
        # If no config file is passed in, use the default values
        config = default_values
    else:
        # Try to open the yaml file with some basic error checking and handling
        config_filepath = os.path.join(project_root, config_filepath)
        config = read_yaml(config_filepath)

    # We make sure that the user has included all necessary keys in the config file
    for default_key in default_values.keys():
        assert (
            default_key in config.keys()
        ), f"Config file must contain key: {default_key}"

    # -- Convert relative paths to absolute paths --
    # Begin with sweep_filename
    if config.sweep_filename:
        abs_path = os.path.join(project_root, config.sweep_filename)
        config.sweep_filename = abs_path
    # Next the model_path. This is a directory where models are stored
    abs_path = os.path.join(project_root, config.model_path)
    config.model_path = abs_path
    # Next the dataset_filename
    abs_path = os.path.join(project_root, "data", config.dataset_filename)
    config.dataset_filename = abs_path
    # Lastly, the test_filenames
    if config.test_filenames:
        for idx, test_filename in enumerate(config.test_filenames):
            abs_path = os.path.join(project_root, "data", "tests", test_filename)
            config.test_filenames[idx] = abs_path
    else:
        config.test_filenames = []

    return config


def read_yaml(filepath):

    assert os.path.exists(filepath), f"YAML file not found: {filepath}"
    # Extract the file extension and check if it is either ".yaml" or ".yml"
    file_extension = os.path.splitext(filepath)[1]
    is_yaml = file_extension in [".yaml", ".yml"]
    assert os.path.isfile(filepath) and is_yaml, "Config file must be a YAML file"

    try:
        # Attempt to load the YAML configuration file
        with open(filepath, "r") as file_handle:
            yaml_file = yaml.safe_load(file_handle) or {}
            # The below is probably dangerous. I don't think the AttrDict constructor
            # appiies recursively. I may need to write a custom function that
            # recursively converts every nested dictionary to an AttrDict
            # Or better yet, just create a general pydantic model to store all of this
            yaml_file = AttrDict(yaml_file)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

    return yaml_file


def create_config(args):

    if isinstance(args, AttrDict) and args.config:
        # This means the config was passed in for a single training run
        print(f"Using config file: {args.config}")
        config = load_config(args.config)
    else:
        # This means no config file was passed in. Use default test values
        print("No configuration file specified. Using default test values")
        config = load_config()

    return config


def validate_config(config):
    """Use this function to check all the keys in the config file and ensure their values are
    valid. This function should raise an error if any of the values are out of range, of an
    invalid type, or otherwise unacceptable or incompatible with model expectations."""

    assert config.pathway in [
        "o2p",
        "p2o",
        "op2op",
    ], "Invalid pathway argument: must be 'o2p', 'p2o', or 'op2op'"

    assert (
        config.d_model % config.nhead == 0
    ), "d_model must be evenly divisible by nhead"

    # Ensure all absolute paths exist and point to correct files or directories
    # Begin with the sweep file
    if config.sweep_filename:
        assert (
            os.path.exists(config.sweep_filename)
        ), f"Sweep file not found: {config.sweep_filename}"

        # Check if the path points to an existing file
        is_file = os.path.isfile(config.sweep_filename)

        # Extract the file extension and check if it is either ".yaml" or ".yml"
        file_extension = os.path.splitext(config.sweep_filename)[1]
        is_yaml_extension = file_extension in [".yaml", ".yml"]

        # Combine the checks in an assert statement
        assert is_file and is_yaml_extension, f"Sweep file must be a YAML file: {config.sweep_filename}"

    # Next the model_path. This is a directory where models are stored
    os.makedirs(config.model_path, exist_ok=True)
    assert os.path.exists(config.model_path), f"Model path not found: {config.model_path}"
    assert (
        os.path.isdir(config.model_path)
    ), f"Model path must be a directory: {config.model_path}"
    # Next the dataset_filename
    assert (
        os.path.exists(config.dataset_filename)
    ), f"Dataset file not found: {config.dataset_filename}"
    assert (os.path.isfile(config.dataset_filename)) and (
        os.path.splitext(config.dataset_filename)[1] == ".csv"
    ), f"Dataset file must be a CSV file: {config.dataset_filename}"

    # Lastly, the test_filenames
    for test_filename in config.test_filenames:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        relative_path = os.path.join(project_root, "data", "tests", test_filename)
        assert os.path.exists(relative_path), f"Test file not found: {relative_path}"
        # Check if the path points to a file
        assert os.path.isfile(relative_path), f"Provided test csv, {relative_path}, is not a file"
        # Check if the file has a .csv extension
        assert os.path.splitext(relative_path)[1] == ".csv", f"Test file must be a CSV file: {relative_path}"

    # Add other checks here as necessary


# ----------------------------------------------------------------------
if __name__ == "__main__":

    # Read in the config arguments. Will be either
    # a config file, or nothing. If nothing is passed in,
    # the program will use hardcoded test arguments
    args = read_args()

    # Next, parse the args and create the config dictionary.
    config = create_config(args)

    # Now check the values in the config dictionary and ensure
    # that they are valid
    validate_config(config)

    print("---Config Values---")
    for k, v in config.items():
        print(f"{k} => {v}")

    # Pass the config dictionary to the main function
    # and begin the training loop
    status = main(config)
    print("Return from main: ", status)
    if status == 0:
        print("Return from main: No errors")
