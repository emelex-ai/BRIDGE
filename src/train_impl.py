from src.wandb_wrapper import WandbWrapper, MyRun
from torch.utils.data import Dataset
from src.model import Model
from attrdict import AttrDict  # DEBUGGING
from typing import List, Tuple, Dict, Any, Union
import random
from pprint import pprint
import torch as pt
import time
import math
import glob
import getpass
import re
import os
import tqdm

# WandbWrapper is a singleton
wandb = WandbWrapper()
run = MyRun()


# ----------------------------------------------------------------------
def single_step(
    gm: Dict[str, Any],
    dataset_slices,
    batch_slice,
    epoch,
    step,
    example_ct,
):
    """
    Docs
    """
    c = gm.c
    ds = gm.ds
    generated_text_table = gm.generated_text_table

    model = gm.model
    device = c.device
    opt = gm.opt

    # print("=================>")
    # print("Learning rate: ", opt.param_groups[0]["lr"])
    # print("c.num_epochs: ", c.num_epochs)

    # for m in model.parameters():
    # print(f"single step, model: {m.requires_grad=}")  # it is True
    # break

    batch = ds[batch_slice]
    # print("batch: ", batch)  # The same on every epoch
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
    if c.pathway in ["op2op", "p2o"]:
        orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
            logits["orth"], orthography["enc_input_ids"][:, 1:]
        )
        loss += orth_loss
    if c.pathway in ["op2op", "o2p"]:
        phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
            logits["phon"], phonology["targets"]
        )
        loss += phon_loss
    # print("phon_loss: ", phon_loss)
    # print("orth_loss: ", orth_loss)
    # print("loss: ", loss)

    # How to disable backward computational and gradient accumulation in the validation loop?
    # https://discuss.pytorch.org/t/how-to-disable-backward-computational-and-gradient-accumulation-in-the-validation-loop/120774

    if model.training:
        # print("TRAIN, global embedding: ", model.state_dict()['global_embedding'])
        loss.backward()
        """
        for k,v in model.state_dict().items():
            print(f"=> state_dict: {k=}: {v.requires_grad= }")
            print(f"=> state_dict: {k=}: {v.grad= }")
            break
        for m in model.parameters():
            print(f"=> param: {m.requires_grad= }")
            print(f"=> param: {m.grad= }")
            break
        """
        opt.step()
        opt.zero_grad()

    # print_weight_norms(model, "weight norms: ")  # Disable when debugging done

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(
        gm,
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
    )
    # metrics = AttrDict({})  # For debugging. Remove. Reinstate about comment later.
    return metrics


# --------------------------------------------------------------------
def calculate_accuracies(pathway, logits, orthography, phonology):
    # --- Calculate Orthographic Accuracy ---
    # Determine model predictions by taking argmax of orthographic logits

    output = {}

    if pathway in ["op2op", "p2o"]:
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

    if pathway in ["op2op", "o2p"]:
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
    gm,
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
):
    example_ct[0] += len(batch["orthography"])
    metrics = {
        f"{gm.mode}/loss": loss,
        f"{gm.mode}/epoch": epoch,
        f"{gm.mode}/example_ct": example_ct[
            0
        ],  # GE: put in a list so I could use it in a function argument
        f"{gm.mode}/global_embedding_magnitude": pt.norm(
            model.global_embedding[0], p=2
        ),
        f"{gm.mode}/model_weights_magnitude": pt.sqrt(
            sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
        ),
        # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
        f"{gm.mode}/generated_text_table": generated_text_table,
    }
    if pathway == "op2op":
        metrics[f"{gm.mode}/orth_loss"] = orth_loss
        metrics[f"{gm.mode}/phon_loss"] = phon_loss

    accuracies = calculate_accuracies(pathway, logits, orthography, phonology)
    for accuracy in accuracies:
        metrics[f"{gm.mode}/{accuracy}"] = accuracies[accuracy]

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
def train_single_epoch(gm, dataset_slices, epoch):  # , single_step_fct):
    c = gm.c
    model = gm.model
    example_ct = [0]
    generated_text_table = gm.generated_text_table

    model.train()
    gm.mode = "train"

    nb_steps = 0
    start = time.time()

    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        metrics[0] = single_step(
            gm, dataset_slices, batch_slice, epoch, step, example_ct
        )
        # metrics[0] = single_step_fct(gm, batch_slice, step, epoch) #, "train")
        nb_steps += 1
        # print("train: nb_steps: ", nb_steps)
        # break  # For debugging only a single step. Remove once debugging complete

    metrics = metrics[0]
    metrics["time_per_train_step"] = (time.time() - start) / nb_steps
    metrics["time_per_train_epoch"] = (
        c.n_steps_per_epoch * metrics["time_per_train_step"]
    )
    return metrics


# ----------------------------------------------------------------------
def validate_single_epoch(gm, dataset_slices, epoch):  # , single_step_fct):
    c = gm.c
    model = gm.model
    example_ct = [0]

    model.eval()
    gm.mode = "val"
    nb_steps = 0
    start = time.time()
    print("==> validation: nb steps: ", len(list(dataset_slices)))

    # Trick to avoid unbounded variables to protect against future Python changes
    # where local variables defined inside loops are not defined upon loop exit.
    # Perform at least one step
    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        metrics[0] = single_step(
            gm, dataset_slices, batch_slice, epoch, step, example_ct
        )
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        nb_steps += 1
        print("valid: nb_steps: ", nb_steps)

    metrics = metrics[0]
    metrics["time_per_val_step"] = (time.time() - start) / nb_steps
    metrics["time_per_val_epoch"] = c.n_steps_per_epoch * metrics["time_per_val_step"]
    return metrics


# ----------------------------------------------------------------------
def load_model(
    model_path, model_id, device="cpu", epochs_completed=-1, continue_training=False
):
    """
    If continue_training=False, a new model is created with incremented model_id.
    Load a model from storage.
    epochs_completed. If < 0, identify file with highest epoch_completed
                      If > 0, use epoch_completed to tag the proper file
    """
    # search all models in model_path/ with model_id
    user = getpass.getuser()
    # remove all characters except 'a-zA-Z'
    user = re.sub(r"[^a-zA-Z]", "", user)
    model_runs = glob.glob(
        model_path + f"/{user}{model_id:05d}*"
    )  # GE added model[0-9]

    if epochs_completed < 0:
        model_id, epochs_completed = get_starting_model_epoch(
            model_path, model_id, continue_training
        )

    model_file_name = get_model_file_name(model_id, epochs_completed)
    model_path = os.path.join(model_path, model_file_name)
    print(f"Load model: {model_path}")

    try:
        chkpt = pt.load(model_path)
    except:
        print("Load failed")
        return None, None, None

    c = chkpt["config"]
    model = chkpt["model"]

    print("==> load, c= ")
    pprint(c)

    gm = AttrDict({})
    gm.c = c
    gm.opt = opt
    gm.model = model
    gm.model_id = model.id
    # Missing from gm: ds, train_dataset_slices, val_dataset_slices, run, generated_text_table

    model_path_load = chkpt["model_path"]  # needed?
    assert model_path == model_path_load

    opt = chkpt["optimizer"]

    return model, opt, c, gm


# ----------------------------------------------------------------------
# Removed epoch and epoch_num
# def save(c, model, opt, model_id):
def save(gm):  # c, model, opt, model_id):
    """
    Save a model to storage.
    """
    # Moving the model to the cpu first increases portability. Otherwise, the model must
    # be reloaded to save GPU it was saved from.  Note that there is an additional transfer cost.
    # model.to('cpu')  # Perhaps this should be controlled via an input parameter

    c = gm.c
    assert "epochs_completed" in c, "'epochs_completed' not in c"
    print("gm.model_id: ", gm.model_id)
    model_file_name = get_model_file_name(gm.model_id, c.epochs_completed)
    model_path = os.path.join(c.model_path, model_file_name)

    # c contains epochs_completed
    pt.save(
        {
            # If model_path is a full path, the model cannot be ported elsewhere
            "model_path": model_path,  # ideally, a full path
            "config": c,
            # No need for model.state_dict if model is saved
            "model": gm.model,  # save all parameters (frozen or not)
            "optimizer": gm.opt,
            "model_id": gm.model_id,
        },
        f=model_path,
    )


# ----------------------------------------------------------------------
def get_starting_model_epoch(model_path, model_id=None, continue_training=False):
    """
    If model_id == None, retrieve the latest model_id.
    If model_id != None: model_id is the desired model_id.
    continue_training == False: find the latest model_id, and increment by 1
    continue_training == True: use model_id to determine the correct run to continue
    """

    # Override the model_id with None if not a continuation run
    if continue_training == False:
        model_id = None

    print("==> enter get_starting_model_epoch: model_id: ", model_id)
    user = getpass.getuser()
    user = re.sub(r"[^a-zA-Z]", "", user)

    def most_recent_model_id_epoch():
        # Called when model_id == None
        print("ENTER most_recent_model_id_epoch")
        # if model_id == None, seek most recent run (run with largest model_id)
        # if model_id is an int, seek the run with the specified model id
        model_runs = glob.glob(model_path + f"/{user}*")
        latest_run = sorted(model_runs)[-1].split("/")[-1]
        print("lastest_run: ", latest_run)
        pattern = r"[a-zA-Z](\d{5})_chkpt(\d{3}).pth"
        match = re.search(pattern, latest_run)
        new_model_id = int(match.group(1))
        print(f"{new_model_id=}")
        epochs_completed = int(match.group(2))
        return new_model_id, epochs_completed

    def most_recent_epoch(model_id):
        model_runs = glob.glob(model_path + f"/{user}{model_id:05d}*")
        try:
            latest_run = sorted(model_runs)[-1].split("/")[-1]
        except:
            print(f"No run found with {model_id=}")
            return -1  # epochs_completed is -1 (error)

        pattern = r"[a-zA-Z]\d{5}_chkpt(\d{3}).pth"
        match = re.search(pattern, latest_run)
        epochs_completed = int(match.group(1))
        return epochs_completed

    if model_id == None:
        print(f"==> most_recent_model_id_epoch, {continue_training=}")
        model_id, epochs_completed = most_recent_model_id_epoch()
        print(f"{model_id=}, {epochs_completed=}")
        # raise "error1"
    else:
        print(f"==> most_recent_epoch, {model_id=}")
        # If a run with `model_id`` cannot be found, return -1
        epochs_completed = most_recent_epoch(model_id)
        print(f"{epochs_completed=}")

    if not continue_training:
        model_id += 1
        epochs_completed = 0

    # print("starting point: model_id, epochs_completed: ", model_id, epochs_completed)
    # print("continue_training: ", continue_training)
    return model_id, epochs_completed


# ----------------------------------------------------------------------
def setup_model(c, ds, num_layers_dict):
    # Continuation run
    if c.continue_training:
        # GE 2023-05-27: fix checkpoint to allow for more general layer structure
        # The code will not work as is.
        # How is epochs_completed known? Via dictionary c.
        file_name = get_model_file_name(model_id, c.epochs_completed)
        chkpt = pt.load(file_name)
        # chkpt = pt.load("f{c.model_path}/model{model_id}_checkpoint{c.epochs_completed}.pth")

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
        print("NEW RUN")
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
        c.epochs_completed = 0
        opt = pt.optim.SGD(model.parameters(), c.learning_rate)
        # opt = pt.optim.AdamW(model.parameters(), c.learning_rate)

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
    norm = pt.sqrt(sum(pt.norm(w[0], p=2) ** 2 for w in model.parameters()))
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
        columns = [f"col{str(i)}" for i in range(m)]
        return wandb.Table(data=A.numpy().tolist(), columns=columns)

    orth_embed_table = fill_table(orth_embed_weights)
    phon_embed_table = fill_table(orth_embed_weights)

    wandb.log(
        {"orth_embed_table": orth_embed_table, "phon_embed_table": phon_embed_table}
    )


# ----------------------------------------------------------------------
def get_model_file_name(model_id, epoch_num):
    user = getpass.getuser()
    return f"{user}{model_id:05d}_chkpt{epoch_num:03d}.pth"


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

    """
    # Better approach
    return all(
        pt.allclose(state_dict1[key], state_dict2[key])
        for key in state_dict1.keys()
    )
    """


# ----------------------------------------------------------------------
def print_model_parameters(model, verbose=False):
    """
    Print model parameters

    Parameters:
    - model: the model whose parameters will be printed
    - verbose: if False, print the first 5 parameters and the first 5 values of model.status_dict()
               if True: print all parameters and value of model.status.dict()
    """

    print("model.parameters: ")
    for i, k in enumerate(model.state_dict()):
        print(k)
        if not verbose and i == 5:
            break

    for i, m in enumerate(model.parameters()):
        norm = pt.norm(m, p=2)
        print(
            f"param[{i}].norm: {norm}, shape: {m.shape}), requires_grad: {m.requires_grad}"
        )
        if not verbose and i == 5:
            break


# ----------------------------------------------------------------------
def run_train_val_loop(gm):
    c = gm.c
    run = gm.run
    train_dataset_slices = gm.train_dataset_slices
    val_dataset_slices = gm.val_dataset_slices
    gm.model.to(c.device)

    metrics: List[Dict] = [{}]
    pbar = create_pbar(c.epochs_completed, c.num_epochs)

    for epoch in pbar:
        metrics[0] = train_single_epoch(
            gm,
            train_dataset_slices,
            epoch,
        )
        more_metrics = validate_single_epoch(
            gm,
            val_dataset_slices,
            epoch,
        )
        if c.max_nb_steps < 0:
            metrics[0].update(more_metrics)
        run.log(metrics[0])
        # Log the embeddings
        log_embeddings(gm.model, gm.ds)
        datum = gm.ds[:1]
        c.epochs_completed = epoch + 1
    save(gm)

    return metrics


# ----------------------------------------------------------------------
def create_pbar(base, nb_iter):
    return tqdm.tqdm(range(base, base + nb_iter), position=0)


# ----------------------------------------------------------------------
def get_device(device=None):
    if not device == None:
        return device

    if pt.cuda.is_available():
        device = pt.device("cuda:0")
    else:
        device = pt.device("cpu")


# ----------------------------------------------------------------------
