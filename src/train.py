from src.wandb_wrapper import WandbWrapper
from src.dataset import ConnTextULDataset
from src.train_impl import get_starting_model_epoch
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple, Dict, Any, Union
import src.train_impl as train_impl
import src.plot_impl as plot_impl
import torch as pt
import tqdm
import random
import math

wandb = WandbWrapper()

# ----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and without hypersweeps
"""


def run_code_sweep(args_dct: Dict):
    # Use startup data to determine starting epoch. Update the model_id
    model_id, epoch_num = get_starting_model_epoch(
        args_dct.model_path, args_dct.continue_training
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
    ds = ConnTextULDataset(
        c, test=c.test, which_dataset=c.which_dataset, nb_rows=c.nb_samples
    )
    results = run_code_impl(run, ds, epoch_num, model_id)

    c.metrics = results
    wandb.finish()
    print("resuts: ", results)
    return c.metrics


# ----------------------------------------------------------------------
def run_code_impl(run, ds, epoch_num, model_id):
    """ """
    print("ENTER run_code_impl")
    c = run.config
    print("run.config: ", run.config)

    MODEL_PATH = c.model_path

    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")

    device = "cpu"

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
    # TODO: call this function from within Main. Use to name the run in wandb.
    # TODO: Store the model name in the config dictionary?

    # Now called from main
    # model_id, epoch_num = train_impl.get_starting_model_epoch(MODEL_PATH, c.continue_training)

    # A number for WandB:
    c.n_steps_per_epoch = len(train_dataset_slices)

    model, opt = train_impl.setup_model(MODEL_PATH, c, ds, num_layers_dict)

    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)
    # run.finish()  # Call not needed for sweeps. SHould be use for non-sweeps.

    # ----------------------------------------------------------------------

    model.to(device)
    # print(f"DEBUG: epoch_num = {epoch_num}, c.epoch_nums = {c.num_epochs}")
    print("c.num_epochs: ", c.num_epochs)
    pbar = tqdm.tqdm(range(epoch_num, epoch_num + c.num_epochs), position=0)
    example_ct = [0]

    # Function closure
    def single_step_fct(batch_slice, step, epoch, mode):
        return train_impl.single_step(
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

    def train_single_epoch_fct(epoch):
        return train_impl.train_single_epoch(
            c,
            model,
            train_dataset_slices,
            epoch,
            single_step_fct,
        )

    def validate_single_epoch_fct(epoch):
        return train_impl.validate_single_epoch(
            c,
            model,
            val_dataset_slices,
            epoch,
            single_step_fct,
        )

    def save_fct(epoch):
        return train_impl.save(epoch, c, model, opt, MODEL_PATH, model_id, epoch_num)

    # generate a type hint for list of dict

    """
    #Preparation for enhanced plotting on wandb

    # Map from the table's columns to the chart's fields
    fields = {"x": "iii", "y": "height", "color": "line_id" }
    data_single = []
    table_single = wandb.Table(data=[], columns=["myepoch", "random", "line_id"])
    # Not sure how this works. Is this the only way to define fields?
    my_custom_line_chart = wandb.plot_table(vega_spec_name="erlebacher/GE_multi_line_plot",
        data_table=table_single, fields=fields)

    # 1) I could create a single table after collecting all the data. 
    # 2) I could log the table every epoch, and then I have multiple tables. 
    # Let us do both
    """
    """
    fields = {"x": "iii", "y": "height", "color": "line_id" }
    data_single = []
    table_single = wandb.Table(data=[], columns=["myepoch", "random", "line_id"])
    # Not sure how this works. Is this the only way to define fields?
    my_custom_line_chart = wandb.plot_table(vega_spec_name="erlebacher/GE_multi_line_plot",
        data_table=table_single, fields=fields)
    """

    metrics: List[Dict] = [{}]

    # ==== OUTER TRAINING LOOP =====
    for epoch in pbar:
        # print("************* epoch: ", epoch, " *******************88")
        metrics[0] = train_single_epoch_fct(epoch)
        more_metrics = validate_single_epoch_fct(epoch)
        if c.max_nb_steps < 0:
            metrics[0].update(more_metrics)
        run.log(metrics[0])
        # Log the embeddings
        train_impl.log_embeddings(model, ds)
        print("Call generate")
        datum = ds[:1]

        output = model.generate(
            c.pathway,
            datum["orthography"]["enc_input_ids"],
            datum["orthography"]["enc_pad_mask"],
            datum["phonology"]["enc_input_ids"],
            datum["phonology"]["enc_pad_mask"],
            deterministic=True,
        )

        print(len(output['orth_probs']))
        print(len(output['orth_tokens']))
        print(len(output['phon_probs']))
        print(len(output['phon_vecs']))
        print(len(output['phon_tokens']))
        print(len(output['global_encoding']))

        save_fct(epoch)

        # Preparation for enhanced plotting on wandb

        """
        # Test Vegalite custom charts
        plot_impl.update_multi_tables(c, epoch, data_single)

        line_plot = wandb.plot.line(
            table_single, x="myepoch", y="random", title="my Line Plot"
        )
        histogram = wandb.plot.histogram(
            table_single, value="loss", title="my Histogram"
        )
        table_single = wandb.Table(data=data_single, columns=["myepoch", "random", "line_id"])
        wandb.log({"table_single": table_single})
        """

    # 🐝 Close wandb
    return metrics[0]


# ----------------------------------------------------------------------
