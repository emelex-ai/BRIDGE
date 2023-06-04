
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

    return more_metrics

#----------------------------------------------------------------------
def single_step(pbar, model, train_dataset_slices, batch_slice, ds, device, example_ct, opt, epoch, step, generated_text_table):
    batch = ds[batch_slice]
    """
    print("type(ds): ", type(ds))
    print("batch_slice: ", batch_slice)
    print("batch = ", batch)
    print("ds: ", ds)
    """
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
    #print("phon_loss: ", phon_loss.item())
    #print("orth_loss: ", orth_loss.item())
    #print("loss: ", loss.item())

    #print_weight_norms(model, "before loss.backward")
    loss.backward()
    opt.step()
    #print_weight_norms(model, "after opt.step")
    opt.zero_grad()

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(logits, orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table)
    #print("DEBUG metrics: ", metrics)
    #raise "error, after print metrics"
    return metrics

#----------------------------------------------------------------------
def compute_metrics(logits,orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table):

    # --- Calculte Orthographic Accuracy ---
    # Determine model predictions by taking argmax of orthographic logits
    orth_pred = pt.argmax(logits["orth"], dim=1)
    # Use the orthographic encoder input ids as labels
    # notice we slice from 1: onwards to remove the start token (not predicted)
    orth_true = orthography["enc_input_ids"][:, 1:]
    # We wish to exclude the padding tokens from the letter-wise accuracy calculation.
    # to ensure that they do not contribute to the loss we find the location of the 4's
    # in orth_true, and set the corresponding locations in orth_pred to -1. This ensures
    # that they will be incorrect, and we can then simply subtract the total number of 
    # 4's from the denominator of the accuracy calculation to exclude them.
    four_locs = pt.where(orth_true == 4) # Find the locations of the padding tokens (4)
    num_fours = (orth_true == 4).sum() # keep track of how many 4's there are
    orth_pred[four_locs] = -1 # Set location of padding tokens to -1
    
    # Create a boolean tensor where each element indicates a correct or incorrect prediction
    # This includes the padding tokens (all padding location tokens are now guaranteed 
    # to be incorrect)
    orth_matches = orth_pred == orth_true 
    letter_wise_accuracy = (orth_matches).sum()/(pt.tensor(orth_pred.shape).prod()-num_fours)

    # To calculate word-wise accuracy, we need to check if all letters in a single word are correct
    # for this calculation, we need to ensure all the padding tokens are correctly predicted so 
    # that their potential falseness does not affect the word-wise accuracy calculation.
    orth_pred[four_locs] = 4 # Set location of padding tokens to 4. Now all padding predictions are correct.
    orth_matches = orth_pred == orth_true 
    # Accuracy is now calculated by checking if all letters in a word are correct, summing, 
    # and dividing by the number of words.
    orth_word_accuracy = (orth_matches).all(dim=1).sum()/orth_pred.shape[0]

    # --- Calculte Phonological Accuracy ---
    # For phonemes, we also need to exclude the padding tokens from the accuracy calculation.
    # However, this time it is more simple because the padding tokens are set to 2, which is 
    # not within the range of predictions for the model, therefore our model will always get the
    # padding tokens wrong. We can simply exclude them from the accuracy calculation by subtracting
    # the number of padding tokens from the denominator of the accuracy calculation.

    # There are three ways to calculate accuracy with the phonological predictions:
    # 1. Phoneme segment accuracy: Accuracy of each phoneme segment across all phonological vectors
    # 2. Phoneme-wise accuracy: Accuracy of each phonological vector across all words
    # 3. Phoneme word accuracy: Accuracy of each word across the entire batch

    # Take argmax of phonological logits to determine model predictions
    phon_pred = pt.argmax(logits["phon"], dim=1)
    phon_true = phonology["targets"]

    # 1. Phoneme segment accuracy: Accuracy of each phoneme segment across all phonological vectors
    phon_matches = (phon_pred == phon_true)
    num_twos = (phon_true == 2).sum()
    phon_segment_accuracy = phon_matches.sum()/(pt.tensor(phon_pred.shape).prod()-num_twos)

    # 2. Phoneme-wise accuracy: Accuracy of each phonological vector across all words
    # Each padding phononological vector is filled with 2's. So we can count the number 
    # of padding phonological vectors by dividing the total number of 2's by the length
    # of the phonological vector.
    num_padding_vectors = num_twos/phon_true.shape[-1]
    # Begin by checking if all phoneme segments are correct along each phonological vector (dim=-1)
    # Then, rather than divide by the product of all dimensions as before, we divide by the product
    # of the batch dimension (num words) and the max_phon_length dimension (num phonological vectors)
    # and subtract the number of padding vectors, because those are guaranteed to be incorrect.
    phoneme_wise_accuracy = (phon_matches).all(dim=-1).sum()/(pt.tensor(phon_pred.shape[:2]).prod()-num_padding_vectors)

    # 3. Phoneme word accuracy: Accuracy of each word across the entire batch
    # To use the .all() method for determining word accuracy we must set all the elements 
    # in phon_pred that correspond to padding in the phon_true, to 2.
    two_locs = pt.where(phon_true == 2) # Find the locations of the padding tokens (2)
    phon_pred[two_locs] = 2 # Set location of padding tokens to 2 (all padding predictions are now correct)
    # Accuracy is now calculated by checking if all phonological vectors in a word are correct, summing, 
    # and dividing by the number of words, i.e. the batch_size.
    phon_matches = (phon_pred == phon_true)
    phon_word_accuracy = (phon_matches).all(dim=-1).all(dim=-1).sum()/phon_pred.shape[0]

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
        "train/letter_wise_accuracy": letter_wise_accuracy,
        "train/orth_word_accuracy": orth_word_accuracy,
        "train/phon_segment_accuracy": phon_segment_accuracy,
        "train/phoneme_wise_accuracy": phoneme_wise_accuracy,
        "train/phon_word_accuracy": phon_word_accuracy,
    }
    #print(f"letter_wise_accuracy: {letter_wise_accuracy}")
    #print(f"orth_word_accuracy: {orth_word_accuracy}")
    #print(f"phon_segment_accuracy: {phon_segment_accuracy}")
    #print(f"phoneme_wise_accuracy: {phoneme_wise_accuracy}")
    #print(f"phon_word_accuracy: {phon_word_accuracy}")


    # DEBUGGING
    #for k,v in metrics.items(): 
        #print(f"metrics[{k}] = {v}")
    
    return metrics
#----------------------------------------------------------------------
def single_epoch(c, model, train_dataset_slices, epoch, single_step_fct):
    model.train()
    nb_steps = 1 
    start = time.time()

    #print("len(train_dataset_slices): ", len(train_dataset_slices))

    #print(f"DEBUG: slice: {train_dataset_slices}, single_epoch: epoch: {epoch}, nb_steps: {nb_steps}, len(train_dataset_slices): {len(train_dataset_slices)}")
    for step, batch_slice in enumerate(train_dataset_slices):
        #print(f"step: {step}, batch_slice: ", batch_slice)
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            #print("max_nb_steps: ", c.max_nb_steps)  # does not reach this point in test mode
            break
        metrics = single_step_fct(batch_slice, step, epoch)   # GE: new
        #print_weight_norms(model, f"DEBUG: step: {step}, norm: ")  # GE: debug
        nb_steps += 1

    #print("nb_steps: ", nb_steps)
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

    # Start a new urn
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
    #print("original model weights")
    #print_weight_norms(model, "initial model weights")
    #print(model)

    #print(
    #    "char/phon tokenizers len: ",
    #    len(ds.character_tokenizer),
    #    len(ds.phonology_tokenizer),
    #)
    return model, opt
#----------------------------------------------------------------------
def create_data_slices(cutpoint, c, ds):
    print(f"DEBUG: cutpoint: {cutpoint}, c.batch_size: {c.batch_size}, len(ds): {len(ds)}")
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

def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")
#----------------------------------------------------------------------
