from src_legacy.wandb_wrapper import WandbWrapper
from src_legacy.dataset import BridgeDataset
from src_legacy.train_impl import get_starting_model_epoch
from typing import List, Tuple, Dict, Any, Union
import src_legacy.train_impl as train_impl
import torch as pt
from addict import Dict as AttrDict


wandb = WandbWrapper()

# ----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and without hypersweeps
"""


def run_code_sweep(args_dct: Dict):
    # Use startup data to determine starting epoch. Update the model_id
    model_id = get_starting_model_epoch()
    wandb_name = train_impl.get_model_file_name(model_id, 0)
    print("sweep: wandb_name: ", wandb_name)
    run = wandb.init(name=wandb_name, config=args_dct)

    config = run.config
    dataset = BridgeDataset(config)
    results = run_code_impl(run, dataset, model_id)

    # config is no longer needed except for possible testing. So no harm done
    # by the next two lines
    config.metrics = results


# ----------------------------------------------------------------------
def run_code(run, model_id):
    config = run.config
    dataset = BridgeDataset(config)
    results = run_code_impl(run, dataset, model_id)
    config.metrics = results
    wandb.finish()
    print("resuts: ", results)
    return config.metrics


# ----------------------------------------------------------------------
def run_code_impl(run, dataset, model_id):
    """ """

    config = run.config

    # Choose automatically if no argument
    device = train_impl.get_device("cpu")

    num_train = int(len(dataset) * config.train_test_split)

    train_dataset_slices, val_dataset_slices = train_impl.create_data_slices(
        num_train, config, dataset
    )

    # Use startup data to determine starting epoch. Update the model_id
    # TODO: call this function from within Main. Use to name the run in wandb.
    # TODO: Store the model name in the config dictionary?

    config.n_steps_per_epoch = len(train_dataset_slices)

    model, opt = train_impl.setup_model(config, dataset)
    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)
    # wandb.watch(model, log="all", log_freq=100)

    # This is a general object containing all necessary parameters, data, and model
    # It will be passed between functions during the training loop. It will also be
    # be used to store intermediate data and the final results.
    gm = AttrDict({})
    gm.cc = config  # We put the entire config file into a single element of this object

    # Next we pack the model, optimizer, and dataset into the object
    gm.model = model
    gm.opt = opt
    gm.dataset = dataset

    # next two lines based on last file saved and continue_training
    gm.model_id = model_id
    gm.epochs_completed = 0

    gm.run = run  # Necessary when using wandb
    # Attributes specific to this model
    gm.train_dataset_slices = train_dataset_slices
    gm.val_dataset_slices = val_dataset_slices
    # Separate table for train and validation data?
    gm.generated_text_table = generated_text_table

    # Not debugged: plots on wandb. I never completed this work.
    # Look in src/plot_impl.py

    # plot_impl.pre_plotting_wandb()
    metrics = train_impl.run_train_val_loop(gm)
    # plot_impl.post_plotting_wandb()  # not debugged

    # 🐝 Close wandb
    return metrics[0], gm


# ----------------------------------------------------------------------
