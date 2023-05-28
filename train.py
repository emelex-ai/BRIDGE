from dataset import ConnTextULDataset
from model import Model
import torch as pt
from wandb_wrapper import WandbWrapper
#import wandb
import tqdm
import sys
import math
import glob
import time
import train_impl as train_impl
import argparse  # NEW LIBRARY (pip install argparse)
#from attrdict import AttrDict  # NEW LIBRARY (pip install attrdict)

#----------------------------------------------------------------------
""" 
More complex code structure to accomodate running wandb with and with hypersweeps
"""
def run_code():
    run = wandb.init()
    run_code_impl(run)

def run_code_impl(run):
    # This approach is prone to error if additional variables are added due to violation of the 
    # DRY principle (Don't Repeat Yourself)
    # When wandb runs in 'disabled' mode, run.config is empty. Thus, it is best to use my wrapper, which will 
    # create a non-empty configuration. I'll have to make sure this configuraation words with the dot notation. 

    print("run.config: ", run.config)
    c = run.config
    num_epochs = c.num_epochs
    CONTINUE = c.CONTINUE
    learning_rate = c.learning_rate
    batch_size = c.batch_size 
    train_test_split = c.train_test_split 
    d_model = c.d_model 
    nhead = c.nhead 
    common_num_layers = c.common_num_layers

    MODEL_PATH = "./models"   # Hardcoded
    ds = ConnTextULDataset()
    cutpoint = int(train_test_split * len(ds))
    
    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")
    
    #device = 'cpu'
    print("device = ", device)
    
    # GE 2023-05-27: added fine-grain control over layer structure
    num_layers_dict = {
        "phon_dec": common_num_layers,
        "phon_enc": common_num_layers,
        "orth_dec": common_num_layers,
        "orth_enc": common_num_layers,
        "mixing_enc": common_num_layers,
    }
    assert d_model % nhead == 0, "d_model must be evenly divisible by nhead"
    
    train_dataset_slices = []
    for batch in range(math.ceil(cutpoint / batch_size)):
        train_dataset_slices.append(
            slice(batch * batch_size, min((batch + 1) * batch_size, cutpoint))
        )
    
    val_dataset_slices = []
    for batch in range(math.ceil((len(ds) - cutpoint) / batch_size)):
        val_dataset_slices.append(
            slice(
                cutpoint + batch * batch_size,
                min(cutpoint + (batch + 1) * batch_size, len(ds)),
            )
        )
    
    # Get latest model run information
    # GE: what if model_runs is True and CONTINUE is False? Shouldn't one go to the else branch?
    model_runs = glob.glob(MODEL_PATH + "/model[0-9]*")  # GE added model[0-9]
    print("model_runs: ", model_runs)
    if model_runs:
        # GE comments
        # Whatever numbering you are using for model_runs, you should use integers with leading zeros, or else sorted will not work correctly.
        # Sorting on letters is dangerous since different people might have different sorting conventions
        latest_run = sorted(model_runs)[-1].split("/")[-1]
        print("latest_run: ", latest_run)  # model_checkpoint35.pth
    
        print("split: ", latest_run.split("_"))
        model_id, epoch_num = int(latest_run.split("_")[0][5:]), int(
            latest_run.split("_")[-1].split(".")[0][10:]
        )
        if not CONTINUE:
            model_id += 1
            epoch_num = 0
    else:
        model_id, epoch_num = 0, 0
    
    # A number for WandB:
    n_steps_per_epoch = len(train_dataset_slices)
    print("n_steps_per_epoch: ", n_steps_per_epoch)
    
    if True:
        if CONTINUE:
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
            opt = pt.optim.AdamW(model.parameters(), learning_rate)
            opt.load_state_dict(chkpt["optimizer"])
        else:
            model = Model(
                len(ds.character_tokenizer),
                len(ds.phonology_tokenizer),
                d_model=d_model,
                nhead=nhead,
                num_layers_dict=num_layers_dict,  # New, GE, 2023-05-27
            )
            opt = pt.optim.AdamW(model.parameters(), learning_rate)
    
        print(
            "char/phon tokenizers len: ",
            len(ds.character_tokenizer),
            len(ds.phonology_tokenizer),
        )
        print("d_model: ", d_model)
        print("n_head: ", nhead)
    
    
        generated_text_table = wandb.Table(columns=["Step", "Generated Output"])
    
        pbar = tqdm.tqdm(range(epoch_num, epoch_num + num_epochs), position=0)
    
        # Does not seem to slow down the code
        run.watch(model, log="all", log_freq=100)  # Comment out by GE
    
        # number steps of epoch: 2223
    
    
        model.to(device)
    
        #----------------------------------------------------------------------
    
        nb_steps_to_run = 2
        example_ct = [0]
    
        for epoch in pbar:
            model.train()
            print("\nTraining Loop...")
            step = 0  # GE: added
            start = time.time()
            for step, batch_slice in enumerate(train_dataset_slices):
                if step >= nb_steps_to_run:  # GE, 2023-05-27
                    end = time.time() 
                    print("time per step (sec); ", (end-start)/nb_steps_to_run)
                    break
                metrics = train_impl.single_step(pbar, model, train_dataset_slices, batch_slice, ds, device, example_ct, opt, epoch, step, generated_text_table)
    
    
            run.log(metrics)  # GE: only execute once per epoch
    
            train_impl.evaluate_model(model, val_dataset_slices, device, opt, ds)
    
        pt.save(
            {
                "epoch": epoch,
                "batch_size": batch_size,
                "d_model": d_model,
                "nhead": nhead,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
            },
            MODEL_PATH + f"/model{model_id}_checkpoint{epoch}.pth",
        )
    
    # 🐝 Close your wandb run
    run.finish()
    
    print("time per step (sec); ", time)
    print("n_steps_per_epoch: ", n_steps_per_epoch)

#----------------------------------------------------------------------
def main():
    """
    """
    #  Three parameters specific to W&B
    entity  = "emelex"
    project = "GE_ConnTextUL"
    is_wandb_enabled = True

    #  Parameters specific to the main code
    num_epochs = 1  # 100
    CONTINUE = False
    learning_rate = 1e-3
    batch_size = 128 # 32
    train_test_split = 0.8
    d_model = 32
    nhead = 4
    common_num_layers = 1

    config = {
        #"starting_epoch": epoch_num,   # Add it back later once code is debugged
        "CONTINUE": CONTINUE, 
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "d_model": d_model,
        "nhead": nhead,
        "learning_rate": learning_rate,
        "train_test_split": train_test_split,
        #"id": model_id,  # Add back later once code is debugged
        "common_num_layers": common_num_layers, 
    }


    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.sweep:
        wandb = WandbWrapper(config=config, is_sweep=True, is_wandb_on=is_wandb_enabled)
        globals().update({'wandb':wandb})
        wandb.login()
        # Is it possible to update a sweep configuration? I'd like the default sweep 
        # configuration to contain the parameters of config. 
        sweep_config = {
            'method' : 'grid',
            'name' : 'sweep_d_model',
            'parameters': {
                'batch_size': {'values': [32, 64]},
                'd_model': {'values': [16, 32]},
                'nlayers': {'values': [1 ]}
            }
        }

        # Update sweep_config with new_params without overwriting existing parameters:
        for param, value in config.items():
            if param not in sweep_config['parameters']:
                sweep_config['parameters'][param] = {'values': [value]}


        sweep_id = wandb.sweep(sweep_config, project=project, entity=entity)
        wandb.agent(sweep_id, run_code)
    else:
        wandb = WandbWrapper(config=config, is_sweep=False, is_wandb_on=is_wandb_enabled)
        globals().update({'wandb':wandb})
        wandb.login()
        # 🐝 initialise a wandb run
        # I created a new project
        run = wandb.init(
            entity=entity,  # Necessary because I am in multiple teams
            project=project,
            config=config,
        )
        # When 'disabled', the returned run.config is an empty dictionary {}
        if not is_wandb_enabled:
            print("wandb is disabled")
            run = wandb.init(config=config)
            # make sure I config is accessible with the dot notation
            #run.config = AttrDict(config)   # INCORRECT
            run_code()
        #if run != None:  # wandb not 'disabled'
        else:
            print("run: ", run)
            print("type(run): ", type(run))
            print("==> run code")
            run_code()


if __name__ == "__main__":
    main()
