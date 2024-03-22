from src.wandb_wrapper import WandbWrapper, MyRun
from src.model import Model
from typing import List, Tuple, Dict, Any, Union
import torch as pt
import time
import math
import getpass
import os
import tqdm
import torch
from datetime import datetime

# WandbWrapper is a singleton
wandb = WandbWrapper()
run = MyRun()


# ----------------------------------------------------------------------
def single_step(
    gm: Dict[str, Any],
    batch_slice,
    epoch,
    step,
    example_ct,
):
    """
    Docs
    """
    dataset = gm.dataset
    generated_text_table = gm.generated_text_table

    model = gm.model
    device = gm.cc.device
    opt = gm.opt

    batch = dataset[batch_slice]
    orthography = batch["orthography"].to(device)
    phonology = batch["phonology"].to(device)

    logits = model(
        gm.cc.pathway,
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
    if gm.cc.pathway in ["op2op", "p2o"]:
        orth_loss = pt.nn.CrossEntropyLoss(ignore_index=4)(
            logits["orth"], orthography["enc_input_ids"][:, 1:]
        )
        loss += orth_loss
    if gm.cc.pathway in ["op2op", "o2p"]:
        phon_loss = pt.nn.CrossEntropyLoss(ignore_index=2)(
            logits["phon"], phonology["targets"]
        )
        loss += phon_loss

    # How to disable backward computational and gradient accumulation in the validation loop?
    # https://discuss.pytorch.org/t/how-to-disable-backward-computational-and-gradient-accumulation-in-the-validation-loop/120774

    if model.training:
        loss.backward()
        opt.step()
        opt.zero_grad()

    metrics = compute_metrics(
        gm,
        gm.cc.pathway,
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
        dataset,
        device,
        model,
        generated_text_table,
    )
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
    dataset,
    device,
    model,
    generated_text_table,
):
    example_ct[0] += len(batch["orthography"])
    metrics = {
        f"{gm.mode}_loss": loss,
        f"{gm.mode}_epoch": epoch,
        f"{gm.mode}_example_ct": example_ct[
            0
        ],  # GE: put in a list so I could use it in a function argument
        f"{gm.mode}_global_embedding_magnitude": pt.norm(
            model.global_embedding[0], p=2
        ),
        f"{gm.mode}_model_weights_magnitude": pt.sqrt(
            sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
        ),
        # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
        f"{gm.mode}_generated_text_table": generated_text_table,
    }
    if pathway == "op2op":
        metrics[f"{gm.mode}_orth_loss"] = orth_loss
        metrics[f"{gm.mode}_phon_loss"] = phon_loss
        phon_pred = pt.argmax(logits["phon"], dim=1)
        phon_true = phonology["targets"]
        metrics["phon_pred"] = phon_pred
        metrics["phon_true"] = phon_true
        mask = torch.all(metrics["phon_true"] == 2, dim=2)  # Convert vectors to 0s
        # print(mask)
        metrics["phon_true"][mask] = 0
        metrics["euc_dis"] = torch.norm(phon_pred.float() - phon_true.float(), dim=2)

    accuracies = calculate_accuracies(pathway, logits, orthography, phonology)
    # distances =calculate_euclideandist(pathway,phon_pred,phon_true)
    for accuracy in accuracies:
        metrics[f"{gm.mode}_{accuracy}"] = accuracies[accuracy]

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
def train_single_epoch(gm, dataset_slices, epoch):

    model = gm.model
    example_ct = [0]

    model.train()
    gm.mode = "train"

    nb_steps = 0
    start = time.time()

    print("train_single_epoch: len(dataset_slices): ", len(dataset_slices))

    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        metrics[0] = single_step(gm, batch_slice, epoch, step, example_ct)
        if gm.cc.max_nb_steps > 0 and nb_steps >= gm.cc.max_nb_steps:
            break
        nb_steps += 1

    metrics = metrics[0]
    metrics["time_per_train_step"] = (time.time() - start) / nb_steps
    metrics["time_per_train_epoch"] = (
        gm.cc.n_steps_per_epoch * metrics["time_per_train_step"]
    )
    return metrics


# ----------------------------------------------------------------------
def validate_single_epoch(gm, dataset_slices, epoch):

    model = gm.model
    example_ct = [0]

    model.eval()
    gm.mode = "valid"

    nb_steps = 0
    start = time.time()
    print("==> validation: nb steps: ", len(list(dataset_slices)))

    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        metrics[0] = single_step(gm, batch_slice, epoch, step, example_ct)
        if gm.cc.max_nb_steps > 0 and nb_steps >= gm.cc.max_nb_steps:
            break
        nb_steps += 1

    metrics = metrics[0]
    metrics["time_per_val_step"] = (time.time() - start) / nb_steps
    metrics["time_per_val_epoch"] = (
        gm.cc.n_steps_per_epoch * metrics["time_per_val_step"]
    )
    return metrics


# ----------------------------------------------------------------------
def save(gm):
    """
    Save a model to storage.
    """
    # Moving the model to the cpu first increases portability. Otherwise, the model must
    # be reloaded to save GPU it was saved from.  Note that there is an additional transfer cost.
    # model.to('cpu')  # Perhaps this should be controlled via an input parameter

    model_file_name = get_model_file_name(gm.model_id, gm.epochs_completed)
    model_path = os.path.join(gm.cc.model_path, model_file_name)

    # I cannot save a wandb.run.config as a pickle file.
    # I simply create an empty dictionary and fill it with the configuration parameters
    # print("----")
    config_wrap = dict(gm.cc)
    # print(f"{type(gm.cc)=}, {gm.cc=}")
    # print(f"{type(dict(gm.cc))=}, {gm.cc=}")
    # print(f"{config_wrap=}")
    # print("----")

    # Remove hooks to allow save
    wandb.unwatch(gm.model)

    # remove gm.run.watch if it exists

    # when using Wandb, I get an error when saving. WHY?

    pt.save(
        {
            # If model_path is a full path, the model cannot be ported elsewhere
            "model_path": model_path,  # ideally, a full path
            "config": config_wrap,  # Perhaps there is no need to save the wandb configuration
            "model_state_dict": gm.model.state_dict(),
            "optimizer_state_dict": gm.opt.state_dict(),
            "model_id": gm.model_id,
        },
        f=model_path,
    )

    wandb.watch(gm.model)

    # gm.cc = cc_bak


# ----------------------------------------------------------------------
def get_starting_model_epoch():

    model_id = get_new_model_id()

    return model_id


# ----------------------------------------------------------------------
def setup_model(config, dataset):
    """ """

    model = Model(config, dataset)
    # opt = pt.optim.SGD(model.parameters(), config.learning_rate)
    opt = pt.optim.AdamW(model.parameters(), config.learning_rate)

    return model, opt


# ----------------------------------------------------------------------
def create_data_slices(cutpoint, config, dataset):
    train_dataset_slices = []
    for batch in range(math.ceil(cutpoint / config.batch_size_train)):
        train_dataset_slices.append(
            slice(
                batch * config.batch_size_train,
                min((batch + 1) * config.batch_size_train, cutpoint),
            )
        )

    # batch size for the validation set is always 1.
    val_dataset_slices = []
    for batch in range(math.ceil((len(dataset) - cutpoint) / config.batch_size_val)):
        val_dataset_slices.append(
            slice(
                cutpoint + batch * config.batch_size_val,
                min(cutpoint + (batch + 1) * config.batch_size_val, len(dataset)),
            )
        )

    return train_dataset_slices, val_dataset_slices


# ----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum(pt.norm(w[0], p=2) ** 2 for w in model.parameters()))
    print(f"==> {msg}, {norm}")


# ----------------------------------------------------------------------
def log_embeddings(model, dataset):
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

    print("wandb.log orth_embed_table: log_embeddings: ", log_embeddings)
    wandb.log(
        {"orth_embed_table": orth_embed_table, "phon_embed_table": phon_embed_table}
    )


# ----------------------------------------------------------------------
def get_user():
    return getpass.getuser().replace("_", "")


# ----------------------------------------------------------------------
def get_new_model_id():
    current_time = datetime.now()
    # Y-m-d is necessary for proper sorting
    date_time = current_time.strftime("%Y-%m-%d_%Hh%Mm")
    seconds = current_time.second + current_time.microsecond / 1000000
    milliseconds = round(seconds * 1000)
    date_time = date_time + f"{milliseconds:05d}ms"
    user = get_user()
    return f"{user}_{date_time}"


# ----------------------------------------------------------------------
def get_model_file_name(model_id, epoch_num):
    # Remove all underscores from the user name
    file_name = f"{model_id}_chkpt{epoch_num:03d}.pth"
    print("before return from model_file_name, file_name: ", file_name)
    return f"{file_name}"


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
    train_dataset_slices = gm.train_dataset_slices
    val_dataset_slices = gm.val_dataset_slices
    gm.model.to(gm.cc.device)

    metrics: List[Dict] = [{}]

    for epoch in range(gm.cc.num_epochs):
        print(f"\n****** {epoch=} *******")
        metrics[0] = train_single_epoch(
            gm,
            train_dataset_slices,
            epoch,
        )

        if val_dataset_slices:
            more_metrics = validate_single_epoch(
                gm,
                val_dataset_slices,
                epoch,
            )

        if gm.cc.max_nb_steps < 0:
            metrics[0].update(more_metrics)
        wandb.log(metrics[0])  # should not be run.log
        # run.log(metrics[0])

        rough_dis = {"\teucleadian_dis": 0}
        # metrics[0].update(rough_dis)
        print("\tmetrics[0]: ", metrics[0])

        # Log the embeddings
        log_embeddings(gm.model, gm.dataset)
        datum = gm.dataset[:1]
        # Apparently, one can only update configuration parameters via the config.update command (2023-09-17)
        # Why does the previous line lead to the error:
        # wandb.sdk.lib.config_util.ConfigError: Attempted to change value of key
        #  If you really want to do this, pass allow_val_change=True to config.update()

        gm.epochs_completed += 1
        if epoch % gm.cc.save_every == 0:
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
def set_seed(config):
    """Set the seed for reproducibility"""
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
