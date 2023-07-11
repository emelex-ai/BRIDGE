from src.wandb_wrapper import WandbWrapper
from src.dataset import ConnTextULDataset
from src.train_impl import get_starting_model_epoch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Dict, Any, Union
import src.train_impl as train_impl
import src.plot_impl as plot_impl
import torch as pt
from addict import Dict as AttrDict
from pprint import pprint

# import tqdm
# import random
# import math

wandb = WandbWrapper()

# ----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and without hypersweeps
"""


def run_code_sweep(args_dct: Dict):
    # Use startup data to determine starting epoch. Update the model_id
    model_id, epoch_num = get_starting_model_epoch(
        args_dct.model_path, model_id=None, continue_training=args_dct.continue_training
    )
    wandb_name = train_impl.get_model_file_name(model_id, epoch_num)
    run = wandb.init(name=wandb_name, config=args_dct)

    c = run.config
    ds = ConnTextULDataset(
        c, test=c.test, which_dataset=c.which_dataset, nb_rows=c.nb_samples
    )
    results = run_code_impl(run, ds, epoch_num, model_id)

    # c is no longer needed except for possible testing. So no harm done
    # by the next two lines
    c.metrics = results


# ----------------------------------------------------------------------
def run_code(run, epoch_num, model_id):
    c = run.config
    print("ENTER run_code")
    ds = ConnTextULDataset(
        c, test=c.test, which_dataset=c.which_dataset, nb_rows=c.nb_samples
    )
    pprint(c)
    results = run_code_impl(run, ds, epoch_num, model_id)

    c.metrics = results
    wandb.finish()
    print("resuts: ", results)
    return c.metrics


# ----------------------------------------------------------------------
def run_code_impl(run, ds, epochs_completed, model_id):
    """ """

    c = run.config

    # Choose automatically if no argument
    device = train_impl.get_device("cpu")

    num_train = int(len(ds) * c.train_test_split)

    train_dataset_slices, val_dataset_slices = train_impl.create_data_slices(
        num_train, c, ds
    )

    # Use startup data to determine starting epoch. Update the model_id
    # TODO: call this function from within Main. Use to name the run in wandb.
    # TODO: Store the model name in the config dictionary?

    c.n_steps_per_epoch = len(train_dataset_slices)

    num_layers_dict = {
        "phon_dec": c.num_layers,
        "phon_enc": c.num_layers,
        "orth_dec": c.num_layers,
        "orth_enc": c.num_layers,
        "mixing_enc": c.num_layers,
    }
    assert c.d_model % c.nhead == 0, "d_model must be evenly divisible by nhead"

    model, opt = train_impl.setup_model(c, ds, num_layers_dict)
    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)

    # Dictionary to store elements of a model that are generic. This could become a class in the future.
    # This dictionary could eventually become a class
    gm = AttrDict({})
    gm.cc = c  # Temporary. gm.c = c has issues. 

    gm.model = model
    gm.opt = opt
    gm.ds = ds
    # next two lines based on last file saved and continue_training
    gm.model_id = model_id
    gm.epochs_completed = epochs_completed 

    gm.run = run  # Necessary when using wandb
    # Attributes specific to this model
    gm.train_dataset_slices = train_dataset_slices
    gm.val_dataset_slices = val_dataset_slices
    # Separate table for train and validation data?
    gm.generated_text_table = generated_text_table

    # plot_impl.pre_plotting_wandb()  # not debugged

    metrics = train_impl.run_train_val_loop(gm)

    # plot_impl.post_plotting_wandb()  # not debugged

    # 🐝 Close wandb
    return metrics[0], gm


# ----------------------------------------------------------------------
