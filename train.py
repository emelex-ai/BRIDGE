from wandb_wrapper import WandbWrapper
from dataset import ConnTextULDataset
import torch as pt
import tqdm
import sys
import time
import train_impl as train_impl
from torch.utils.data import DataLoader, Subset

wandb = WandbWrapper()

# ----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and without hypersweeps
"""


def run_code():
    run = wandb.init()
    ds = ConnTextULDataset(test=False, which_dataset=run.config.which_dataset)
    run_code_impl(run, ds)


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
    #print("device = ", device)

    # GE 2023-05-27: added fine-grain control over layer structure
    num_layers_dict = {
        "phon_dec": c.num_layers,
        "phon_enc": c.num_layers,
        "orth_dec": c.num_layers,
        "orth_enc": c.num_layers,
        "mixing_enc": c.num_layers,
    }
    assert c.d_model % c.nhead == 0, "d_model must be evenly divisible by nhead"

    num_train = int(len(ds) * c.train_test_split)
    #cutpoint = int(c.train_test_split * len(ds))

    train_dataset_slices, val_dataset_slices = train_impl.create_data_slices(
        num_train, c, ds
    )
    print("c = ", c)
    #print("train_dataset_slices = ", train_dataset_slices)
    #print("val_dataset_slices = ", val_dataset_slices)
    print("len(train_dataset_slices) = ", len(train_dataset_slices))
    print("len(val_dataset_slices) = ", len(val_dataset_slices))

    # Use startup data to determine starting epoch. Update the model_id
    model_id, epoch_num = train_impl.get_starting_model_epoch(MODEL_PATH, c)

    # A number for WandB:
    c.n_steps_per_epoch = len(train_dataset_slices)

    model, opt = train_impl.setup_model(MODEL_PATH, c, ds, num_layers_dict)

    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)  # Comment out by GE

    # ----------------------------------------------------------------------

    model.to(device)
    #print(f"DEBUG: epoch_num = {epoch_num}, c.epoch_nums = {c.num_epochs}")
    pbar = tqdm.tqdm(range(epoch_num, epoch_num + c.num_epochs), position=0)
    example_ct = [0]

    # Function closure
    single_step_fct = lambda batch_slice, step, epoch: train_impl.single_step(
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
    )
    single_epoch_fct = lambda epoch: train_impl.single_epoch(
        c, model, train_dataset_slices, epoch, single_step_fct, 
    )
    evaluate_model_fct = lambda: train_impl.evaluate_model(
        model, val_dataset_slices, device, opt, ds, 
    )
    save_fct = lambda epoch: train_impl.save(
        epoch, c, model, opt, MODEL_PATH, model_id, epoch_num
    )

    for epoch in pbar:
        #print("epoch = ", epoch)
        metrics = single_epoch_fct(epoch)
<<<<<<< HEAD
        run.log(metrics)  
=======
>>>>>>> ac4a6299f3970d6da76f7d4f52bb947bab7a4e0b
        # When experimenting, skip evaluate_model_fct for speedup
        # if c.max_nb_steps == -1, run all steps and evaluate the model
        if c.max_nb_steps < 0:
            metrics.update(evaluate_model_fct())

        run.log(metrics)
        save_fct(epoch)

    # 🐝 Close your wandb run
    run.finish()

    ##print("time per step (sec); ", time)
    #print("n_steps_per_epoch: ", c.n_steps_per_epoch)

# ----------------------------------------------------------------------
