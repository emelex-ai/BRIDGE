from src.wandb_wrapper import WandbWrapper
from src.dataset import ConnTextULDataset
import src.train_impl as train_impl
import torch as pt
import tqdm
import sys
import time
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Dict, Any, Union

wandb = WandbWrapper()

# ----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and without hypersweeps
"""


def run_code():
    run = wandb.init()
    c = run.config
    ds = ConnTextULDataset(test=c.test, which_dataset=c.which_dataset, nb_rows=c.nb_samples)
    return run_code_impl(run, ds)


def run_code_impl(run, ds):
    """ """
    c = run.config
    print("run.config: ", run.config)

    MODEL_PATH = c.model_path

    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")

    device = 'cpu'

    num_layers_dict = {
        "phon_dec": c.num_layers,
        "phon_enc": c.num_layers,
        "orth_dec": c.num_layers,
        "orth_enc": c.num_layers,
        "mixing_enc": c.num_layers,
    }
    assert c.d_model % c.nhead == 0, "d_model must be evenly divisible by nhead"

    num_train = int(len(ds) * c.train_test_split)

    train_dataset_slices, val_dataset_slices = train_impl.create_data_slices(
        num_train, c, ds
    )

    # Use startup data to determine starting epoch. Update the model_id
    model_id, epoch_num = train_impl.get_starting_model_epoch(MODEL_PATH, c)

    # A number for WandB:
    c.n_steps_per_epoch = len(train_dataset_slices)

    model, opt = train_impl.setup_model(MODEL_PATH, c, ds, num_layers_dict)

    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)  

    # ----------------------------------------------------------------------

    model.to(device)
    #print(f"DEBUG: epoch_num = {epoch_num}, c.epoch_nums = {c.num_epochs}")
    pbar = tqdm.tqdm(range(epoch_num, epoch_num + c.num_epochs), position=0)
    example_ct = [0]

    # Function closure
    single_step_fct = lambda batch_slice, step, epoch, mode: train_impl.single_step(
        c,
        pbar,
        model,
        train_dataset_slices,
        batch_slice,
        ds,
        device,
        opt,
        epoch,
        step,
        generated_text_table,
        example_ct,
        mode,
    )
    train_single_epoch_fct = lambda epoch: train_impl.train_single_epoch(
        c, model, train_dataset_slices, epoch, single_step_fct, 
    )
    validate_single_epoch_fct = lambda epoch: train_impl.validate_single_epoch(
        c, model, val_dataset_slices, epoch, single_step_fct,
    )
    save_fct = lambda epoch: train_impl.save(
        epoch, c, model, opt, MODEL_PATH, model_id, epoch_num
    )

# generate a type hint for list of dict

    metrics: List[Dict] = [{}]

    # ==== OUTER TRAINING LOOP =====
    for epoch in pbar:
        print("************* epoch: ", epoch, " *******************88")
        metrics[0] = train_single_epoch_fct(epoch)
        print("type(metrics): ", type(metrics))
        more_metrics = validate_single_epoch_fct(epoch)
        if c.max_nb_steps < 0:
            metrics[0].update(more_metrics)
        run.log(metrics[0])
        # Log the embeddings
        train_impl.log_embeddings(model, ds)
        print("generate")
        train_impl.generate(ds, device)
        save_fct(epoch)

    # 🐝 Close wandb 
    run.finish()
    return metrics[0]

# ----------------------------------------------------------------------
