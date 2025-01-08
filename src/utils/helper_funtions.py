from datetime import datetime
import getpass
import os

import torch
import random
import numpy as np


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_date():
    return datetime.now().date().strftime("%Y_%m_%d")


def get_user():
    return getpass.getuser().replace("_", "")


def get_run_name(model_artifacts_dir="model_artifacts"):
    """
    Generate a unique run name based on the current date. If the date has changed, restart numbering from 001;
    otherwise, increment the last run number.

    :param model_artifacts_dir: Directory to store experiment runs.
    :return: A unique run name.
    """
    # Get today's date in the desired format
    current_date = datetime.now().strftime("%Y_%m_%d")

    # Ensure the base directory exists
    if not os.path.exists(model_artifacts_dir):
        os.makedirs(model_artifacts_dir)

    # List all directories matching the current date
    existing_runs = [
        d
        for d in os.listdir(model_artifacts_dir)
        if os.path.isdir(os.path.join(model_artifacts_dir, d)) and d.startswith(current_date)
    ]

    # Extract run numbers for the current date
    run_numbers = [int(d.split("_")[-1]) for d in existing_runs if d.split("_")[-1].isdigit()]

    # Determine the next run number
    next_run_number = max(run_numbers, default=0) + 1

    # Generate the new run name
    run_name = f"{current_date}_run_{next_run_number:03d}"

    # Create the directory for the new run
    run_path = os.path.join(model_artifacts_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    return run_name


<<<<<<< HEAD
def get_model_file_name(model_id: str, epoch_num: int):
    file_name = f"{model_id}_chkpt{epoch_num:03d}.pth"
    return f"{file_name}"


def setup_new_run():
    model_id = get_new_model_id()
    model_file_name = get_model_file_name(model_id, 0)
    return model_id, model_file_name


def handle_model_continuation(training_config):
    ##############################
    #  This function will be used to handle the scenario where we wish
    #  to continue training an existing model. This is not yet implemented.
    ##############################

    # TODO: Add logic here to handle continuation of a model

    # For now, setup new simulation with new model_id

    model_id, model_file_name = setup_new_run()
    if not training_config.model_id:
        training_config.model_id = model_id

    print(f"New run: {model_file_name}")

    return model_id, model_file_name


=======
>>>>>>> main
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
<<<<<<< HEAD
        torch.cuda.manual_seed_all(seed)
=======
        torch.cuda.manual_seed_all(seed)
>>>>>>> main
