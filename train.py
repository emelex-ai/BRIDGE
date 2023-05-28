from wandb_wrapper import WandbWrapper
from dataset import ConnTextULDataset
import torch as pt
import tqdm
import sys
import math
import time
import train_impl as train_impl

wandb = WandbWrapper()

#----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and with hypersweeps
"""
def run_code():
    run = wandb.init()
    run_code_impl(run)

def run_code_impl(run):
    # This approach of copying variables from the dictionary is prone to error if additional variables are 
    # added due to violation of the 
    # DRY principle (Don't Repeat Yourself)
    # Unless variables are in a type loop, I recommend using the dictionary. 

    c = run.config

    MODEL_PATH = "./models"   # Hardcoded
    ds = ConnTextULDataset()
    cutpoint = int(c.train_test_split * len(ds))
    
    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")
    
    #device = 'cpu'
    print("device = ", device)
    
    # GE 2023-05-27: added fine-grain control over layer structure
    num_layers_dict = {
        "phon_dec": c.common_num_layers,
        "phon_enc": c.common_num_layers,
        "orth_dec": c.common_num_layers,
        "orth_enc": c.common_num_layers,
        "mixing_enc": c.common_num_layers,
    }
    assert c.d_model % c.nhead == 0, "d_model must be evenly divisible by nhead"
    
    train_dataset_slices = []
    for batch in range(math.ceil(cutpoint / c.batch_size)):
        train_dataset_slices.append(
            slice(batch * c.batch_size, min((batch + 1) * c.batch_size, cutpoint))
        )
    
    val_dataset_slices = []
    for batch in range(math.ceil((len(ds) - cutpoint) / c.batch_size)):
        val_dataset_slices.append(
            slice(
                cutpoint + batch * c.batch_size,
                min(cutpoint + (batch + 1) * c.batch_size, len(ds)),
            )
        )
    
    # Use startup data to determine starting epoch. Update the model_id
    model_id, epoch_num = train_impl.get_starting_model_epoch(MODEL_PATH, c)
    
    # A number for WandB:
    n_steps_per_epoch = len(train_dataset_slices)
    print("n_steps_per_epoch: ", n_steps_per_epoch)
    
    model, opt = train_impl.setup_model(MODEL_PATH, c, ds, num_layers_dict)

    """
    def setup_model(MODEL_PATH, ds):
        # Continuation run
        if c.CONTINUE:
            # GE 2023-05-27: fix checkpoint to allow for more general layer structure
            # The code will not work as is. 
            chkpt = pt.load(MODEL_PATH + f"/model{model_id}_checkpoint{epoch_num}.pth")
            # GE: TODO:  Construct a layer dictionary from the chekpointed data 
            model = Model(
                len(ds.character_tokenizer),
                len(ds.phonology_tokenizer),
                d_model=chkpt["d_model"],
                nhead=chkpt["nhead"],
            )
            model.load_state_dict(chkpt["model"])
            opt = pt.optim.AdamW(model.parameters(), c.learning_rate)
            opt.load_state_dict(chkpt["optimizer"])
    
        # Start a new urn
        else:
            model = Model(
                len(ds.character_tokenizer),
                len(ds.phonology_tokenizer),
                d_model=c.d_model,
                nhead=c.nhead,
                num_layers_dict=num_layers_dict,  # New, GE, 2023-05-27
            )
            opt = pt.optim.AdamW(model.parameters(), c.learning_rate)
    
        print(
            "char/phon tokenizers len: ",
            len(ds.character_tokenizer),
            len(ds.phonology_tokenizer),
        )
        return model, opt
    """

    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    run.watch(model, log="all", log_freq=100)  # Comment out by GE

    #----------------------------------------------------------------------

    model.to(device)
    pbar = tqdm.tqdm(range(epoch_num, epoch_num + c.num_epochs), position=0)
    nb_steps_to_run = 2
    example_ct = [0]

    # Function closure (GE)
    # I may have forgotten one or two arguments. Must be checked. 
    single_step_fct = lambda batch_slice, step, epoch : train_impl.single_step(pbar, model, train_dataset_slices, batch_slice, ds, device, example_ct, opt, epoch, step, generated_text_table)
    evaluate_model_fct = lambda : train_impl.evaluate_model(model, val_dataset_slices, device, opt, ds)
    single_epoch_fct = lambda epoch : train_impl.single_epoch(model, train_dataset_slices, nb_steps_to_run, epoch, single_step_fct)
    save_fct = lambda epoch : train_impl.save(epoch, c, model, opt, MODEL_PATH, model_id, epoch_num)

    for epoch in pbar:
        metrics = single_epoch_fct(epoch)
        run.log(metrics)  # GE: only execute once per epoch
        evaluate_model_fct() # GE: 05/28
        save_fct(epoch)

    # 🐝 Close your wandb run
    run.finish()
    
    print("time per step (sec); ", time)
    print("n_steps_per_epoch: ", n_steps_per_epoch)

#----------------------------------------------------------------------
