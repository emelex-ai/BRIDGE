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
    # This approach of copying variables from the dictionary is prone to error if additional variables are
    # added due to violation of the
    # DRY principle (Don't Repeat Yourself)
    # Unless variables are in a type loop, I recommend using the dictionary.

    c = run.config

    MODEL_PATH = c.model_path
    # GE, May 30, 2024: added nb_rows argument to read a subset
    #ds = ConnTextULDataset() # first sweep, use all rows
    #ds = ConnTextULDataset(nb_rows=1000) # first sweep, use `nb_rows` rows
    #ds = ConnTextULDataset(nb_rows=10000) # first sweep
    # Extract the top 1000 rows and return a Dataset
    # For some reason, Subset does not work. It should. 
    #ds = Subset(ds, indices=range(len(ds)))  # Generates an error. Something wrong. 
    # ds = ds[slice(0, 1000, 1)]  # len(ds) returns 2. Why is that? 

    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")

    device = 'cpu'
    #print("device = ", device)

    # GE 2023-05-27: added fine-grain control over layer structure
    num_layers_dict = {
        "phon_dec": c.common_num_layers,
        "phon_enc": c.common_num_layers,
        "orth_dec": c.common_num_layers,
        "orth_enc": c.common_num_layers,
        "mixing_enc": c.common_num_layers,
    }
    assert c.d_model % c.nhead == 0, "d_model must be evenly divisible by nhead"

    num_train = int(len(ds) * c.train_test_split)
    #cutpoint = int(c.train_test_split * len(ds))

    """ MIGHT ADD LATER. Not sure why you are not using a DataLoader """
    """
    # Added by Gordon
    dataset_train = train_impl.ConnDataset(ds[:num_train])
    dataset_valid = train_impl.ConnDataset(ds[num_train:])
    #dataset_train = train_impl.ConnDataset(ds[:num_train])
    #dataset_valid = train_impl.ConnDataset(ds[num_train:])

    # The advantage of dataloaders is that it is very easy to change batch size from 
    # epoch to epoch
    loader_train = DataLoader(ds, batch_size=c.batch_size, shuffle=True)
    loader_valid = DataLoader(ds, batch_size=1, shuffle=True)
    """

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

    # Function closure (GE)
    # I may have forgotten one or two arguments. Must be checked.
    single_step_fct = lambda batch_slice, step, epoch: train_impl.single_step(
        pbar,
        model,
        train_dataset_slices,
        batch_slice,
        ds,
        device,
        example_ct,
        opt,
        epoch,
        step,
        generated_text_table,
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
        # When experimenting, skip evaluate_model_fct for speedup
        if c.max_nb_steps < 0:
            metrics.update(evaluate_model_fct())

        run.log(metrics)
        save_fct(epoch)

    # 🐝 Close your wandb run
    run.finish()

    ##print("time per step (sec); ", time)
    #print("n_steps_per_epoch: ", c.n_steps_per_epoch)


# ----------------------------------------------------------------------
