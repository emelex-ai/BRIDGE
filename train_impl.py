
import torch as pt
from wandb_wrapper import WandbWrapper, MyRun
from model import Model
import time
import math
import glob
from torch.utils.data import Dataset

# WandbWrapper is a singleton
wandb = WandbWrapper()
run = MyRun()


# GE: 2023-05-27: simplify code
def evaluate_model(model, val_dataset_slices, device, opt, ds):
    model.eval()  # GE: eval once per epoch
    with pt.no_grad():
        print("\nValidation Loop...")
        for step, batch_slice in enumerate(val_dataset_slices):
            batch = ds[batch_slice]
            orthography, phonology = batch["orthography"].to(device), batch[
                "phonology"
            ].to(device)
            logits = model(
                orthography["enc_input_ids"],
                orthography["enc_pad_mask"],
                orthography["dec_input_ids"],
                orthography["dec_pad_mask"],
                phonology["enc_input_ids"],
                phonology["enc_pad_mask"],
                phonology["dec_input_ids"],
                phonology["dec_pad_mask"],
            )

            val_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
                logits["orth"], orthography["enc_input_ids"][:, 1:]
            )
            val_loss = val_loss + pt.nn.CrossEntropyLoss(ignore_index=2)(
                logits["phon"], phonology["targets"]
            )
            more_metrics = {"val/val_loss": val_loss}
            run.log(more_metrics)

#----------------------------------------------------------------------
def single_step(pbar, model, train_dataset_slices, batch_slice, ds, device, example_ct, opt, epoch, step, generated_text_table):
    batch = ds[batch_slice]
    #print("batch_slice: ", batch_slice)
    #print("batch = ", batch)
    orthography, phonology = batch["orthography"].to(device), batch[
        "phonology"
    ].to(device)
    logits = model(
        orthography["enc_input_ids"],
        orthography["enc_pad_mask"],
        orthography["dec_input_ids"],
        orthography["dec_pad_mask"],
        phonology["enc_input_ids"],
        phonology["enc_pad_mask"],
        phonology["dec_input_ids"],
        phonology["dec_pad_mask"],
    )

    orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
        logits["orth"], orthography["enc_input_ids"][:, 1:]
    )
    phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
        logits["phon"], phonology["targets"]
    )
    loss = orth_loss + phon_loss
    print("phon_loss: ", phon_loss.item())
    print("orth_loss: ", orth_loss.item())
    print("loss: ", loss.item())

    print_weight_norms(model, "before loss.backward")
    loss.backward()
    opt.step()
    print_weight_norms(model, "after opt.step")
    opt.zero_grad()

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(logits, orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table)
    print("DEBUG metrics: ", metrics)
    #raise "error, after print metrics"
    return metrics

#----------------------------------------------------------------------
def compute_metrics(logits,orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table):
    # NEW NEW NEW NEW NEW NEW NEW NEW NEW

    # Accuracy function for orthography.
    # Take argmax of orthographic logits for accuracy comparison:
    A_orth = pt.argmax(logits["orth"], dim=1)
    # Keep orthographic encoder input ids unchanged:
    B_orth = orthography["enc_input_ids"][:, 1:]
    # Compute orthographic accuracy:
    word_accuracy = pt.tensor(pt.where((A_orth == B_orth).all(dim=1))[0].size())/pt.tensor(A_orth.size())[0]
    char_accuracy = (A_orth == B_orth).sum()/pt.tensor(A_orth.shape).prod()

    #orth_accuracy = 
        #pt.tensor(pt.where((A_orth == B_orth).all(dim=1))[0].size())
        #/ pt.tensor(A_orth.size())[0]
    #)

    # Accuracy function for phonology.
    # Compute dimensions for phonological logit and target reshaping:
    oldshape = logits["phon"].size()
    newshape_0 = oldshape[0] * oldshape[2]
    newshape_1 = oldshape[3]
    # Take argmax of phonological logits for accuracy comparison, reshaping to get a square tensor:
    A_phon = pt.argmax(logits["phon"], dim=1)
    A_phon = pt.reshape(A_phon, [newshape_0, newshape_1])
    print(f"\n{epoch}/{step}: A_phon shape:", A_phon.size())  # GE: changed
    # Reshape phonological targets:
    B_phon = phonology["targets"]
    B_phon = pt.reshape(B_phon, [newshape_0, newshape_1])
    print(f"{epoch}/{step}: B_phon shape:", B_phon.size(), "\n")  # GE: changed
    # Compute phonoloigcal accuracy:
    phon_accuracy = (
        pt.tensor(pt.where((A_phon == B_phon).all(dim=1))[0].size())
        / pt.tensor(A_phon.size())[0]
    )

    # END END END END END END END END END

    # Now we generate orthographic tokens and phonological vectors for the input 'elephant'
    if 1:
        orth = ds.character_tokenizer.encode(["elephant"])
        orthography = orth["enc_input_ids"].to(device)
        orthography_mask = orth["enc_pad_mask"].to(device)
        phon = ds.phonology_tokenizer.encode(["elephant"])
        phonology = [
            [t.to(device) for t in tokens] for tokens in phon["enc_input_ids"]
        ]
        phonology_mask = phon["enc_pad_mask"].to(device)

        # GE:  I do not understand
        generation = model.generate(
            orthography, orthography_mask, phonology, phonology_mask
        )

        generated_text = ds.character_tokenizer.decode(
            generation["orth"].tolist()
        )[0]
        # Log the text in the WandB table
        generated_text_table.add_data(step, generated_text)

    example_ct[0] += len(batch["orthography"])
    metrics = {
        "train/orth_loss": orth_loss,
        "train/phon_loss": phon_loss,
        "train/train_loss": loss,
        "train/epoch": epoch,
        "train/example_ct": example_ct[0],  # GE: put in a list so I could use it in a function argument
        "train/global_embedding_magnitude": pt.norm(
            model.global_embedding[0], p=2
        ),
        "train/model_weights_magnitude": pt.sqrt(
            sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
        ),
        # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
        "train/generated_text_table": generated_text_table,
        "word accuracy": word_accuracy,
        "phonological accuracy": phon_accuracy,
        "character accuracy": char_accuracy,
    }

    # DEBUGGING
    #for k,v in metrics.items(): 
        #print(f"metrics[{k}] = {v}")

    
    return metrics
#----------------------------------------------------------------------
def single_epoch(c, model, train_dataset_slices, epoch, single_step_fct):
    model.train()
    nb_steps = 0  # GE: added
    start = time.time()

    print("len(train_dataset_slices): ", len(train_dataset_slices))


    for step, batch_slice in enumerate(train_dataset_slices):
        print(f"step: {step}, batch_slice: ", batch_slice)
        nb_steps += 1
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            print("max_nb_steps: ", c.max_nb_steps)  # does not reach this point in test mode
            raise "should not occur"
            break
        metrics = single_step_fct(batch_slice, step, epoch)   # GE: new
        print_weight_norms(model, f"DEBUG: step: {step}, norm: ")  # GE: debug

    print("nb_steps: ", nb_steps)
    metrics['time_per_step'] = (time.time() - start) / nb_steps
    metrics['time_per_epoch'] = c.n_steps_per_epoch * metrics['time_per_step']
    return metrics
#----------------------------------------------------------------------
def save(epoch, c, model, opt, MODEL_PATH, model_id, epoch_num):
    pt.save(
        {
            "epoch": epoch,
            "batch_size": c.batch_size,
            "d_model": c.d_model,
            "nhead": c.nhead,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            # "epoch_num": epoch_num,   # GE: added (recommended): starting epoch. 
                                        # Good for self-consistency check. 
        },
        # GE: pay attention: I replaced epoch by c.epoch (c is configuration)
        MODEL_PATH + f"/model{model_id}_checkpoint{epoch}.pth",  # HARDCODED
    )
#----------------------------------------------------------------------
def get_starting_model_epoch(path, c):
    # Get latest model run information
    # GE: what if model_runs is True and CONTINUE is False? Shouldn't one go to the else branch?
    # What if you want to  start a new run? Must you empty the folder?
    # c.continue: if True, continuation run. 
    #             if False, start a new run and update the model_id
    model_runs = glob.glob(path + "/model[0-9]*")  # GE added model[0-9]

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
        if not c.CONTINUE:
            model_id += 1
            epoch_num = 0
    else:
        model_id, epoch_num = 0, 0

    return model_id, epoch_num
#----------------------------------------------------------------------
def setup_model(MODEL_PATH, c, ds, num_layers_dict):
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
    print("original model weights")
    print_weight_norms(model, "initial model weights")
    print(model)

    print(
        "char/phon tokenizers len: ",
        len(ds.character_tokenizer),
        len(ds.phonology_tokenizer),
    )
    return model, opt
#----------------------------------------------------------------------
def create_data_slices(cutpoint, c, ds):
    # GE: question: why does ds[batch_slice] look like a dictionary?
    print("create_data_slices: cutpoint: ", cutpoint)
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

    """
    for slice_ in train_dataset_slices: 
        print("train slice: ", slice_)
    for slice_ in val_dataset_slices: 
        print("valid slice: ", slice_)
    raise "error slice"
    """

    """ GE additions """
    """ NOT USED 
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=5, shuffle=True)
    for step in loader: 
        print("type(step): ", type(step))
    raise "end"
    print("end of create_data_slices")
    dss = ds[train_dataset_slices[0]]
    print("type(ds[slice[0:3]]): ", type(ds[slice[0:3]]))
    print("slice: ", train_dataset_slices[0])
    print("type(dss): ", type(dss))  # dict
    print("keys(dss): ", list(keys(dss)))
    print("type(dss[0]): ", type(dss[0]))  # dict
    raise "end of create_data_slices"
    """

    """ GE: original code """
    print("len slices: ", len(train_dataset_slices), len(val_dataset_slices))
    return train_dataset_slices, val_dataset_slices
#----------------------------------------------------------------------
# Not used
class ConnDataset(Dataset):
    def __init__(self, ds):
        self.data = ds

    def __len__(self):
        return len(ds)

    def __get_item(self, i):
        return ds[i]

#----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")
#----------------------------------------------------------------------
