from src_legacy.wandb_wrapper import WandbWrapper
from src_legacy.dataset import BridgeDataset

# from src.train_impl import get_starting_model_epoch
# from torch.utils.data import DataLoader, Subset
# from typing import List, Tuple, Dict, Any, Union
# import src.train_impl as train_impl
# import torch as pt
# import tqdm
import random
import math

wandb = WandbWrapper()


# ----------------------------------------------------------------------
def update_multi_tables(c, epoch, data_single):
    # Demonstration of custom charts with W&B
    # New random line every epoch
    if not c.wandb:
        return

    # Create a table with the columns to plot (add to the table each epoch)
    # Define table . Fill it multiple times during the run.
    data_multi = []
    offset = 2 * epoch
    for i in range(20):
        # Compute the data and dump to the table at the end of each epoch
        data_multi.append(  # three columns. Adding "line_id" is a hack.
            [i, random.random() + math.log(1 + i) + offset + random.random(), epoch]
        )
        # Accumulate all the data, dump to the table at the end of the simulation
        data_single.append(  # three columns. Adding "line_id" is a hack.
            [i, random.random() + math.log(1 + i) + offset + random.random(), epoch]
        )

    table_multi = wandb.Table(data=data_multi, columns=["myepoch", "random", "line_id"])
    wandb.log({"epoch_random_multi": table_multi})

    # use the table to populate various custom charts
    line_plot = wandb.plot.line(
        table_multi, x="myepoch", y="random", title="my Line Plot"
    )
    histogram = wandb.plot.histogram(table_multi, value="loss", title="my Histogram")

    # Log custom tables, which will show up as custommizable charts in the W&B UI
    wandb.log({"line_1": line_plot, "histogram_1": histogram})


# ----------------------------------------------------------------------
"""
    def pre_plotting_wandb():
        #Preparation for enhanced plotting on wandb

        # Map from the table's columns to the chart's fields
        fields = {"x": "iii", "y": "height", "color": "line_id" }
        data_single = []
        table_single = wandb.Table(data=[], columns=["myepoch", "random", "line_id"])
        # Not sure how this works. Is this the only way to define fields?
        my_custom_line_chart = wandb.plot_table(vega_spec_name="erlebacher/GE_multi_line_plot",
            data_table=table_single, fields=fields)

        # 1) I could create a single table after collecting all the data. 
        # 1) I could log the table every epoch, and then I have multiple tables. 
        # Let us do both
        
        fields = {"x": "iii", "y": "height", "color": "line_id" }
        data_single = []
        table_single = wandb.Table(data=[], columns=["myepoch", "random", "line_id"])
        # Not sure how this works. Is this the only way to define fields?
        my_custom_line_chart = wandb.plot_table(vega_spec_name="erlebacher/GE_multi_line_plot",
            data_table=table_single, fields=fields)
"""
# ----------------------------------------------------------------------
"""
    def post_plotting_wandb():
        output = model.generate(
            c.pathway,
            datum["orthography"]["enc_input_ids"],
            datum["orthography"]["enc_pad_mask"],
            datum["phonology"]["enc_input_ids"],
            datum["phonology"]["enc_pad_mask"],
            deterministic=True,
        )

        print(len(output["orth_probs"]))
        print(len(output["orth_tokens"]))
        print(len(output["phon_probs"]))
        print(len(output["phon_vecs"]))
        print(len(output["phon_tokens"]))
        print(len(output["global_encoding"]))

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
