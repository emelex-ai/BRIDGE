from src.wandb_wrapper import WandbWrapper, MyRun
from torch.utils.data import Dataset
from src.model import Model
from attrdict import AttrDict  # DEBUGGING
import random
import torch as pt
import time
import math
import glob
import getpass
import re
import os

# WandbWrapper is a singleton
wandb = WandbWrapper()
run = MyRun()


def evaluate_model(pathway, model, val_dataset_slices, device, opt, ds, mode):
    model.eval()

    with pt.no_grad():
        # print("\nValidation Loop...")
        for step, batch_slice in enumerate(val_dataset_slices):
            batch = ds[batch_slice]
            orthography, phonology = batch["orthography"].to(device), batch[
                "phonology"
            ].to(device)
            logits = model(
                pathway,
                orthography["enc_input_ids"],
                orthography["enc_pad_mask"],
                orthography["dec_input_ids"],
                orthography["dec_pad_mask"],
                phonology["enc_input_ids"],
                phonology["enc_pad_mask"],
                phonology["dec_input_ids"],
                phonology["dec_pad_mask"],
            )

            loss = 0
            orth_loss = None
            phon_loss = None
            if pathway == "op2op" or pathway == "p2o":
                orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
                    logits["orth"], orthography["enc_input_ids"][:, 1:]
                )
                loss += orth_loss
            if pathway == "op2op" or pathway == "o2p":
                phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
                    logits["phon"], phonology["targets"]
                )
                loss += phon_loss

            more_metrics = {mode + "/loss": loss}
            if pathway == "op2op":
                more_metrics[mode + "/orth_loss"] = orth_loss
                more_metrics[mode + "/phon_loss"] = phon_loss

    return more_metrics


# ----------------------------------------------------------------------
def single_step(
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
):
    """ """
    print("Learning rate: ", opt.param_groups[0]["lr"])

    batch = ds[batch_slice]
    orthography = batch["orthography"].to(device)
    phonology = batch["phonology"].to(device)
    # In an ideal world, a DataLoader should be used:
    # then the call to the model could be: model(dataloader.next())  (pseudocode)
    # GE: I do not understand why one inputs decoder input ids into the model
    logits = model(
        c.pathway,
        orthography["enc_input_ids"],
        orthography["enc_pad_mask"],
        orthography["dec_input_ids"],
        orthography["dec_pad_mask"],
        phonology["enc_input_ids"],
        phonology["enc_pad_mask"],
        phonology["dec_input_ids"],
        phonology["dec_pad_mask"],
    )

    loss = 0
    orth_loss = None
    phon_loss = None
    if c.pathway == "op2op" or c.pathway == "p2o":
        orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
            logits["orth"], orthography["enc_input_ids"][:, 1:]
        )
        loss += orth_loss
    if c.pathway == "op2op" or c.pathway == "o2p":
        phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
            logits["phon"], phonology["targets"]
        )
        loss += phon_loss
    print("phon_loss: ", phon_loss)
    print("orth_loss: ", orth_loss)
    print("loss: ", loss)

    # How to disable backward computational and gradient accumulation in the validation loop?
    # https://discuss.pytorch.org/t/how-to-disable-backward-computational-and-gradient-accumulation-in-the-validation-loop/120774

    if model.training:
        loss.backward()
        opt.step()
        opt.zero_grad()

    print_weight_norms(model, "weight norms: ")  # Disable when debugging done

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(
        c.pathway,
        logits,
        orthography,
        phonology,
        batch,
        example_ct,
        orth_loss,
        phon_loss,
        loss,
        epoch,
        step,
        ds,
        device,
        model,
        generated_text_table,
        mode,
    )
    #metrics = AttrDict({})  # For debugging. Remove. Reinstate about comment later. 
    return metrics


# --------------------------------------------------------------------
def calculate_accuracies(pathway, logits, orthography, phonology):
    # --- Calculate Orthographic Accuracy ---
    # Determine model predictions by taking argmax of orthographic logits

    output = {}

    if pathway == "op2op" or pathway == "p2o":
        orth_pred = pt.argmax(logits["orth"], dim=1)
        # Use the orthographic encoder input ids as labels
        # notice we slice from 1: onwards to remove the start token (not predicted)
        orth_true = orthography["enc_input_ids"][:, 1:]

        # Create a mask for valid positions (not padding)
        orth_valid_mask = orth_true != 4

        # Apply the mask to true and predicted values
        masked_orth_true = orth_true[orth_valid_mask]
        masked_orth_pred = orth_pred[orth_valid_mask]

        # Calculate letter-wise accuracy
        correct_matches = (masked_orth_pred == masked_orth_true).sum()
        letter_wise_accuracy = correct_matches.float() / orth_valid_mask.sum().float()

        # To calculate word-wise accuracy, we need to check if all letters in a single word are correct
        orth_pred[~orth_valid_mask] = 4
        word_wise_mask = orth_pred == orth_true
        orth_word_accuracy = word_wise_mask.all(dim=1).float().mean()
        output["letter_wise_accuracy"] = letter_wise_accuracy
        output["word_wise_accuracy"] = orth_word_accuracy

    if pathway == "op2op" or pathway == "o2p":
        # --- Calculate Phonological Accuracy ---
        phon_pred = pt.argmax(logits["phon"], dim=1)
        phon_true = phonology["targets"]

        # Create a mask for valid positions (not padding)
        phon_valid_mask = phon_true != 2

        # Apply the mask to true and predicted values
        masked_phon_true = phon_true[phon_valid_mask]
        masked_phon_pred = phon_pred[phon_valid_mask]

        # Phoneme segment accuracy
        correct_phoneme_segments = (masked_phon_pred == masked_phon_true).sum()
        phon_segment_accuracy = (
            correct_phoneme_segments.float() / phon_valid_mask.sum().float()
        )

        # Phoneme-wise accuracy
        phoneme_wise_mask = phon_pred == phon_true
        phoneme_wise_accuracy = phoneme_wise_mask.all(dim=-1).sum() / (
            masked_phon_true.shape[0] / phon_true.shape[-1]
        )

        # Phoneme word accuracy
        word_accuracies = [
            word[target != 2].all().int()
            for word, target in zip(phoneme_wise_mask, phon_true)
        ]
        # phon_word_accuracy = phoneme_wise_mask.all(dim=-1).all(dim=-1).sum()/phon_true.shape[0]
        phon_word_accuracy = sum(word_accuracies) / len(word_accuracies)
        output["phon_segment_accuracy"] = phon_segment_accuracy
        output["phoneme_wise_accuracy"] = phoneme_wise_accuracy
        output["phon_word_accuracy"] = phon_word_accuracy

    return output


# ----------------------------------------------------------------------
def compute_metrics(
    pathway,
    logits,
    orthography,
    phonology,
    batch,
    example_ct,
    orth_loss,
    phon_loss,
    loss,
    epoch,
    step,
    ds,
    device,
    model,
    generated_text_table,
    mode,
):
    example_ct[0] += len(batch["orthography"])
    metrics = {
        mode + "/loss": loss,
        mode + "/epoch": epoch,
        mode
        + "/example_ct": example_ct[
            0
        ],  # GE: put in a list so I could use it in a function argument
        mode + "/global_embedding_magnitude": pt.norm(model.global_embedding[0], p=2),
        mode
        + "/model_weights_magnitude": pt.sqrt(
            sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
        ),
        # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
        mode + "/generated_text_table": generated_text_table,
    }
    if pathway == "op2op":
        metrics[mode + "/orth_loss"] = orth_loss
        metrics[mode + "/phon_loss"] = phon_loss

    accuracies = calculate_accuracies(pathway, logits, orthography, phonology)
    for accuracy in accuracies:
        metrics[mode + "/" + accuracy] = accuracies[accuracy]

    return metrics


# ----------------------------------------------------------------------
def average_metrics_over_epoch(all_metrics):
    # Calculate metric averages over one epoch
    metrics = all_metrics[0]
    # Average each key over all metrics in all_metrics
    for m in all_metrics[1:]:
        for k in m:
            try:
                metrics[k] += m[k]
            except:
                pass
    for k in metrics:
        try:
            metrics[k] /= len(all_metrics)
        except:
            pass

    return metrics


# ----------------------------------------------------------------------
def train_single_epoch(c, model, dataset_slices, epoch, single_step_fct):
    example_ct = [0]

    model.train()
    nb_steps = 0
    start = time.time()

    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        metrics[0] = single_step_fct(batch_slice, step, epoch, "train")
        nb_steps += 1
        print("train: nb_steps: ", nb_steps)

    metrics = metrics[0]
    metrics["time_per_train_step"] = (time.time() - start) / nb_steps
    metrics["time_per_train_epoch"] = (
        c.n_steps_per_epoch * metrics["time_per_train_step"]
    )
    return metrics


# ----------------------------------------------------------------------
def validate_single_epoch(c, model, dataset_slices, epoch, single_step_fct):
    example_ct = [0]

    model.eval()
    nb_steps = 0
    start = time.time()
    print("==> validation: nb steps: ", len(list(dataset_slices)))

    # Trick to avoid unbounded variables to protect against future Python changes
    # where local variables defined inside loops are not defined upon loop exit.
    # Perform at least one step
    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        metrics[0] = single_step_fct(batch_slice, step, epoch, "val")
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        nb_steps += 1
        print("valid: nb_steps: ", nb_steps)

    metrics = metrics[0]
    metrics["time_per_val_step"] = (time.time() - start) / nb_steps
    metrics["time_per_val_epoch"] = c.n_steps_per_epoch * metrics["time_per_val_step"]
    return metrics


# ----------------------------------------------------------------------
def load_model(model_path, model_id, epoch_num, device='cpu'):
    """
    Load a model from storage.
    """
    print("==> enter load_model: ", model_path)
    model_file_name = get_model_file_name(model_id, epoch_num)
    model_path = os.path.join(model_path, model_file_name)

    chkpt = pt.load(model_path)

    c = chkpt["config"]
    model = chkpt["model"]
    model.to(device)  # IS THIS NEEDED if on 'cpu'? 

    # for k, v in model.__dict__.items():
    # print("load k,v= ", k, v)

    epoch = chkpt["epoch"]  # needed?
    model_path_load = chkpt["model_path"]  # needed?
    assert model_path == model_path_load

    # Reconstruct optimizer
    # opt_type = chkpt["optimizer_type"]
    # print("load, opt_type: ", opt_type)
    # opt = opt_type(model.parameters())   # <<< ERROR
    opt = chkpt["optimizer"]

    # for k, v in opt.__dict__.items():
    #     print("==> load k,v opt: ", k, v)
    #for k, v in opt.state_dict().items():
        #print("==> load k,v opt.state_dict(): ", k, v)

    # Update optimizer parameters
    # opt_state_dict = chkpt["optimizer_state_dict"]
    # opt.load_state_dict(opt_state_dict)

    # Import to return the model to the device. Make sure this happens after loading
    # the optimizer. 
    #model.to(device)  # Perhaps this should be controlled via an input parameter

    return model, opt, c


# ----------------------------------------------------------------------
def save(epoch, c, model, opt, MODEL_PATH, model_id, epoch_num):
    """
    Save a model to storage.
    """
    # Moving the model to the cpu first increases portability. Otherwise, the model must
    # be reloaded to save GPU it was saved from.  Note that there is an additional transfer cost. 
    #model.to('cpu')  # Perhaps this should be controlled via an input parameter

    model_file_name = get_model_file_name(model_id, epoch_num)
    model_path = os.path.join(MODEL_PATH, model_file_name)
    # print("model_path: ", model_path)

    # for k, v in opt.__dict__.items():
    #     print("==> save k,v opt: ", k, v)
    #for k, v in opt.state_dict().items():
        #print("==> save k,v opt.state_dict(): ", k, v)

    # for k, v in model.__dict__.items():
    # print("save k,v= ", k, v)

    # print("save: ", model_path)
    pt.save(
        {
            "model_path": model_path,  # ideally, a full path
            "config": c,
            "epoch": epoch,
            # No need for model.state_dict
            "model": model,  # save all parameters (frozen or not)
            # "optimizer_state_dict": opt.state_dict(),
            # "optimizer_type": type(opt),
            "optimizer": opt,
        },
        model_path,
    )
    print("==> return from save")


# ----------------------------------------------------------------------
def get_starting_model_epoch(path, continue_training):
    # Get latest model run information
    # GE: what if model_runs is True and CONTINUE is False? Shouldn't one go to the else branch?
    # What if you want to  start a new run? Must you empty the folder?
    # c.continue: if True, continuation run.
    #             if False, start a new run and update the model_id
    user = getpass.getuser()
    user = re.sub(r"[^a-zA-Z]", "", user)
    model_runs = glob.glob(path + f"/{user}[0-9]*")  # GE added model[0-9]

    if model_runs:
        # GE comments
        # Whatever numbering you are using for model_runs, you should use
        # integers with leading zeros, or else sorted will not work correctly.
        # Sorting on letters is dangerous since different people might have different sorting conventions
        latest_run = sorted(model_runs)[-1].split("/")[-1]
        pattern = r"[a-zA-Z](\d{3})_chkpt(\d{3}).pth"
        match = re.search(pattern, latest_run)
        model_id = int(match.group(1))
        epoch_num = int(match.group(2))
        # epoch_num = latest_run.split("_")[-1].split(".")[0][10:]
        # model_id = int(latest_run.split("_")[0][5:])

        if not continue_training:
            model_id += 1
            epoch_num = 0
            print("NOT CONTINUE TRAINING, new model_id: ", model_id)
    else:
        model_id = 0
        epoch_num = 0

    print("starting point: model_id, epoch_num: ", model_id, epoch_num)
    print("continue_training: ", continue_training)
    return model_id, epoch_num


# ----------------------------------------------------------------------
def setup_model(MODEL_PATH, c, ds, num_layers_dict):
    # Continuation run
    if c.continue_training:
        # GE 2023-05-27: fix checkpoint to allow for more general layer structure
        # The code will not work as is.
        chkpt = pt.load(MODEL_PATH + f"/model{model_id}_checkpoint{epoch_num}.pth")
        # GE: TODO:  Construct a layer dictionary from the chekpointed data

        model = Model(
            len(ds.character_tokenizer),
            len(ds.phonology_tokenizer),
            d_model=chkpt["d_model"],
            nhead=chkpt["nhead"],
            max_orth_seq_len=ds.max_orth_seq_len,
            max_phon_seq_len=ds.max_phon_seq_len,
            num_layers_dict=num_layers_dict,
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
        # opt = pt.optim.AdamW(model.parameters(), c.learning_rate)
        opt = pt.optim.SGD(model.parameters(), c.learning_rate)

    return model, opt


# ----------------------------------------------------------------------
def create_data_slices(cutpoint, c, ds):
    train_dataset_slices = []
    for batch in range(math.ceil(cutpoint / c.batch_size_train)):
        train_dataset_slices.append(
            slice(
                batch * c.batch_size_train,
                min((batch + 1) * c.batch_size_train, cutpoint),
            )
        )

    # batch size for the validation set is always 1.
    val_dataset_slices = []
    for batch in range(math.ceil((len(ds) - cutpoint) / c.batch_size_val)):
        val_dataset_slices.append(
            slice(
                cutpoint + batch * c.batch_size_val,
                min(cutpoint + (batch + 1) * c.batch_size_val, len(ds)),
            )
        )

    return train_dataset_slices, val_dataset_slices


# ----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")


# ----------------------------------------------------------------------
def log_embeddings(model, ds):
    """
    Extract the embedding layer from the model: one for orthography, one for phonology.
    This is used to visualize the embeddings in wandb
    """

    orth_embedding_layer = model.orthography_embedding
    phon_embedding_layer = model.phonology_embedding
    orth_embed_weights = orth_embedding_layer.weight.detach().data
    phon_embed_weights = phon_embedding_layer.weight.detach().data
    print("orth_embed_weights shape: ", orth_embed_weights.shape)
    print("phon_embed_weights shape: ", phon_embed_weights.shape)
    print("phon_embed_weights type: ", type(phon_embed_weights))
    # wandb.log({"hist_embeddings": [wandb.Histogram(orth_embed_weights), wandb.Histogram(phon_embed_weights)]})
    # log the embeddings as two tables: one for orthography, one for phonology

    def fill_table(A):
        n, m = A.numpy().shape
        columns = ["col" + str(i) for i in range(m)]
        return wandb.Table(data=A.numpy().tolist(), columns=columns)

    orth_embed_table = fill_table(orth_embed_weights)
    phon_embed_table = fill_table(orth_embed_weights)

    wandb.log(
        {"orth_embed_table": orth_embed_table, "phon_embed_table": phon_embed_table}
    )


# ----------------------------------------------------------------------
def get_model_file_name(model_id, epoch_num):
    user = getpass.getuser()
    return f"{user}{model_id:03d}_chkpt{epoch_num:03d}.pth"


# ----------------------------------------------------------------------
def compare_state_dicts(state_dict1, state_dict2):
    # First, compare if they have the same keys
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Then, compare the values (tensors) for each key
    for key in state_dict1.keys():
        if not pt.allclose(state_dict1[key], state_dict2[key]):
            return False

    # If all keys and values match, the state_dicts are the same
    return True
