
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


def evaluate_model(model, val_dataset_slices, device, opt, ds):
    model.eval()  
    with pt.no_grad():
        #print("\nValidation Loop...")
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

            orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
                logits["orth"], orthography["enc_input_ids"][:, 1:]
            )
            phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
                logits["phon"], phonology["targets"]
            )
            more_metrics = {"val/loss": orth_loss + phon_loss,
                            "val/orth_loss": orth_loss,
                            "val/phon_loss": phon_loss,}
            run.log(more_metrics)

#----------------------------------------------------------------------
def single_step(c, pbar, model, train_dataset_slices, batch_slice, ds, device, opt, epoch, step, generated_text_table, example_ct):
    """ """
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

    orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
        logits["orth"], orthography["enc_input_ids"][:, 1:]
    )
    phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
        logits["phon"], phonology["targets"]
    )
    loss = orth_loss + phon_loss

    loss.backward()
    opt.step()
    opt.zero_grad()

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(logits, orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table)
    return metrics

#----------------------------------------------------------------------
def compute_metrics(logits,orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table):

    # Accuracy function for orthography.
    # Determine model preditcions by taking argmax of orthographic logits
    orth_pred = pt.argmax(logits["orth"], dim=1)
    # Keep orthographic encoder input ids unchanged as labels
    # notice we slice from 1: onwards to remove the start token
    orth_true = orthography["enc_input_ids"][:, 1:]
    # Compute orthographic accuracy:
    #orth_word_accuracy = pt.tensor(pt.where((A_orth == B_orth).all(dim=1))[0].size())/pt.tensor(A_orth.size())[0]
    orth_word_accuracy = (orth_pred == orth_true).all(dim=1).sum()/orth_pred.shape[0]
    letter_wise_accuracy = (orth_pred == orth_true).sum()/pt.tensor(orth_pred.shape).prod()

    # Accuracy function for phonology.
    # Compute dimensions for phonological logit and target reshaping:
    oldshape = logits["phon"].size()
    newshape_0 = oldshape[0] * oldshape[2]
    newshape_1 = oldshape[3]
    # Take argmax of phonological logits for accuracy comparison, reshaping to get a square tensor:
    phon_pred = pt.argmax(logits["phon"], dim=1)
    phon_pred = pt.reshape(phon_pred, [newshape_0, newshape_1])
    #print(f"\n{epoch}/{step}: phon_pred shape:", phon_pred.size())  # GE: changed
    # Reshape phonological targets:
    phon_true = phonology["targets"]
    phon_true = pt.reshape(phon_true, [newshape_0, newshape_1])
    #print(f"{epoch}/{step}: phon_true shape:", phon_true.size(), "\n")  # GE: changed
    # Compute phonoloigcal accuracy:
    phon_word_accuracy = (
        pt.tensor(pt.where((phon_pred == phon_true).all(dim=1))[0].size())
        / pt.tensor(phon_pred.size())[0]
    )
    phoneme_wise_accuracy = (phon_pred == phon_true).sum()/pt.tensor(A_orth.shape).prod()

    # END END END END END END END END END

    # Now we generate orthographic tokens and phonological vectors for the input 'elephant'
    if 0:
        orth = ds.character_tokenizer.encode(["elephant"])
        orthography = orth["enc_input_ids"].to(device)
        orthography_mask = orth["enc_pad_mask"].to(device)
        phon = ds.phonology_tokenizer.encode(["elephant"])
        phonology = [
            [t.to(device) for t in tokens] for tokens in phon["enc_input_ids"]
        ]
        phonology_mask = phon["enc_pad_mask"].to(device)

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
        "train/letter_wise_accuracy": letter_wise_accuracy,
        "train/orth_word_accuracy": orth_word_accuracy,
        "train/phoneme_wise_accuracy": phoneme_wise_accuracy,
        "train/phon_word_accuracy": phon_word_accuracy,
    }


    return metrics
#----------------------------------------------------------------------
def single_epoch(c, model, train_dataset_slices, epoch, single_step_fct):
    example_ct = [0]

    model.train()
    nb_steps = 1 
    start = time.time()

    for step, batch_slice in enumerate(train_dataset_slices):
        #print(f"step: {step}, batch_slice: ", batch_slice)
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            #print("max_nb_steps: ", c.max_nb_steps)  # does not reach this point in test mode
            break
        metrics = single_step_fct(batch_slice, step, epoch)   # GE: new
        print_weight_norms(model, f"DEBUG: step: {step}, norm: ")  # GE: debug
        nb_steps += 1

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
        #print("latest_run: ", latest_run)  # model_checkpoint35.pth
        #print("split: ", latest_run.split("_"))
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
            max_seq_len=ds.max_seq_len,
            num_layers_dict=num_layers_dict,  # New, GE, 2023-05-27
            d_embedding=c.d_embedding,
        )
        model.load_state_dict(chkpt["model"])
        opt = pt.optim.AdamW(model.parameters(), c.learning_rate)
        opt.load_state_dict(chkpt["optimizer"])

    # Start a new run
    else:
        model = Model(
            len(ds.character_tokenizer),
            len(ds.phonology_tokenizer),
            d_model=c.d_model,
            d_embedding=c.d_embedding,
            max_orth_seq_len=ds.max_orth_seq_len,
            max_phon_seq_len=ds.max_phon_seq_len,
            nhead=c.nhead,
            num_layers_dict=num_layers_dict,  # New, GE, 2023-05-27
        )
        opt = pt.optim.AdamW(model.parameters(), c.learning_rate)

    return model, opt
#----------------------------------------------------------------------
def create_data_slices(cutpoint, c, ds):
    #print(f"DEBUG: cutpoint: {cutpoint}, c.batch_size: {c.batch_size}, len(ds): {len(ds)}")
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

    return train_dataset_slices, val_dataset_slices

#----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")
#----------------------------------------------------------------------
