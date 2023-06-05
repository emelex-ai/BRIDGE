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


def evaluate_model(model, val_dataset_slices, device, opt, ds, mode):
    model.eval()

    with pt.no_grad():
        # print("\nValidation Loop...")
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
            more_metrics = {
                mode + "/loss": orth_loss + phon_loss,
                mode + "/orth_loss": orth_loss,
                mode + "/phon_loss": phon_loss,
            }

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
    batch = ds[batch_slice]
    orthography, phonology = batch["orthography"].to(device), batch["phonology"].to(
        device
    )
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

    # How to disable backward computational and gradient accumulation in the validation loop?
    # https://discuss.pytorch.org/t/how-to-disable-backward-computational-and-gradient-accumulation-in-the-validation-loop/120774

    if model.training:
        loss.backward()
        opt.step()
        opt.zero_grad()

    # Suggestion: compute cheap metrics every step, more complex metrics every epoch
    metrics = compute_metrics(
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
    return metrics


# ----------------------------------------------------------------------
def compute_metrics(
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
    four_locs = pt.where(orth_true == 4)  # Find the locations of the padding tokens (4)
    num_fours = (orth_true == 4).sum()  # keep track of how many 4's there are
    orth_pred[four_locs] = -1  # Set location of padding tokens to -1

    # Create a boolean tensor where each element indicates a correct or incorrect prediction
    # This includes the padding tokens (all padding location tokens are now guaranteed
    # to be incorrect)
    orth_matches = orth_pred == orth_true
    letter_wise_accuracy = (orth_matches).sum() / (
        pt.tensor(orth_pred.shape).prod() - num_fours
    )

    # To calculate word-wise accuracy, we need to check if all letters in a single word are correct
    # for this calculation, we need to ensure all the padding tokens are correctly predicted so
    # that their potential falseness does not affect the word-wise accuracy calculation.
    orth_pred[
        four_locs
    ] = 4  # Set location of padding tokens to 4. Now all padding predictions are correct.
    orth_matches = orth_pred == orth_true
    # Accuracy is now calculated by checking if all letters in a word are correct, summing,
    # and dividing by the number of words.
    orth_word_accuracy = (orth_matches).all(dim=1).sum() / orth_pred.shape[0]

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
    phon_matches = phon_pred == phon_true
    num_twos = (phon_true == 2).sum()
    phon_segment_accuracy = phon_matches.sum() / (
        pt.tensor(phon_pred.shape).prod() - num_twos
    )

    # 2. Phoneme-wise accuracy: Accuracy of each phonological vector across all words
    # Each padding phononological vector is filled with 2's. So we can count the number
    # of padding phonological vectors by dividing the total number of 2's by the length
    # of the phonological vector.
    num_padding_vectors = num_twos / phon_true.shape[-1]
    # Begin by checking if all phoneme segments are correct along each phonological vector (dim=-1)
    # Then, rather than divide by the product of all dimensions as before, we divide by the product
    # of the batch dimension (num words) and the max_phon_length dimension (num phonological vectors)
    # and subtract the number of padding vectors, because those are guaranteed to be incorrect.
    phoneme_wise_accuracy = (phon_matches).all(dim=-1).sum() / (
        pt.tensor(phon_pred.shape[:2]).prod() - num_padding_vectors
    )

    # 3. Phoneme word accuracy: Accuracy of each word across the entire batch
    # To use the .all() method for determining word accuracy we must set all the elements
    # in phon_pred that correspond to padding in the phon_true, to 2.
    two_locs = pt.where(phon_true == 2)  # Find the locations of the padding tokens (2)
    phon_pred[
        two_locs
    ] = 2  # Set location of padding tokens to 2 (all padding predictions are now correct)
    # Accuracy is now calculated by checking if all phonological vectors in a word are correct, summing,
    # and dividing by the number of words, i.e. the batch_size.
    phon_matches = phon_pred == phon_true
    phon_word_accuracy = (phon_matches).all(dim=-1).all(dim=-1).sum() / phon_pred.shape[
        0
    ]

    # Now we generate orthographic tokens and phonological vectors for the input 'elephant'
    if 0:
        orth = ds.character_tokenizer.encode(["elephant"])
        orthography = orth["enc_input_ids"].to(device)
        orthography_mask = orth["enc_pad_mask"].to(device)
        phon = ds.phonology_tokenizer.encode(["elephant"])
        phonology = [[t.to(device) for t in tokens] for tokens in phon["enc_input_ids"]]
        phonology_mask = phon["enc_pad_mask"].to(device)

        generation = model.generate(
            orthography, orthography_mask, phonology, phonology_mask
        )

        generated_text = ds.character_tokenizer.decode(generation["orth"].tolist())[0]
        # Log the text in the WandB table
        generated_text_table.add_data(step, generated_text)

    example_ct[0] += len(batch["orthography"])
    metrics = {
        mode + "/orth_loss": orth_loss,
        mode + "/phon_loss": phon_loss,
        mode + "/train_loss": loss,
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
        mode + "/letter_wise_accuracy": letter_wise_accuracy,
        mode + "/orth_word_accuracy": orth_word_accuracy,
        mode + "/phon_segment_accuracy": phon_segment_accuracy,
        mode + "/phoneme_wise_accuracy": phoneme_wise_accuracy,
        mode + "/phon_word_accuracy": phon_word_accuracy,
    }
    # print(f"letter_wise_accuracy: {letter_wise_accuracy}")
    # print(f"orth_word_accuracy: {orth_word_accuracy}")
    # print(f"phon_segment_accuracy: {phon_segment_accuracy}")
    # print(f"phoneme_wise_accuracy: {phoneme_wise_accuracy}")
    # print(f"phon_word_accuracy: {phon_word_accuracy}")

    return metrics


# ----------------------------------------------------------------------
def average_metrics_over_epoch(all_metrics):
    # Calculate metric averages over one epoch
    metrics = all_metrics[0]
    # Average each key over all metrics in all_metrics
    for m in all_metrics[1:]:
        for k in m:
            metrics[k] += m[k]
    for k in metrics:
        metrics[k] /= len(all_metrics)

    return metrics

# ----------------------------------------------------------------------
def train_single_epoch(c, model, dataset_slices, epoch, single_step_fct):
    example_ct = [0]

    model.train()
    nb_steps = 0
    all_metrics = []

    print("==> train_single_epoch: nb steps: ", len(list(dataset_slices)))

    start = time.time()
    for step, batch_slice in enumerate(dataset_slices):
        print("training step: ", step)
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        metrics = single_step_fct(batch_slice, step, epoch, "train")
        all_metrics.append(metrics)
        nb_steps += 1

    metrics = average_metrics_over_epoch(all_metrics)
    metrics["time_per_val_step"] = (time.time() - start) / nb_steps
    metrics["time_per_val_epoch"] = c.n_steps_per_epoch * metrics["time_per_val_step"]
    return metrics

# ----------------------------------------------------------------------
def validate_single_epoch(c, model, dataset_slices, epoch, single_step_fct):
    example_ct = [0]

    model.eval()
    nb_steps = 0
    all_metrics = []

    print("==> validation: nb steps: ", len(list(dataset_slices)))

    # Trick to avoid unbounded variables to protect against future Python changes
    # where local variables defined inside loops are not defined upon loop exit.
    # Perform at least one step

    start = time.time()
    for step, batch_slice in enumerate(dataset_slices):
        print("validation step: ", step)
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        metrics = single_step_fct(batch_slice, step, epoch, "val")
        all_metrics.append(metrics)
        nb_steps += 1

    metrics = average_metrics_over_epoch(all_metrics)
    metrics["time_per_val_step"] = (time.time() - start) / nb_steps
    metrics["time_per_val_epoch"] = c.n_steps_per_epoch * metrics["time_per_val_step"]
    return metrics

# ----------------------------------------------------------------------
def save(epoch, c, model, opt, MODEL_PATH, model_id, epoch_num):
    # Has to be fixed to reflect the additional parameters in the configuration (June 4, 2023)
    pt.save(
        {
            "epoch": epoch,
            "batch_size_train": c.batch_size_train,
            "batch_size_val": c.batch_size_val,
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


# ----------------------------------------------------------------------
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
        # print("latest_run: ", latest_run)  # model_checkpoint35.pth
        # print("split: ", latest_run.split("_"))
        model_id, epoch_num = int(latest_run.split("_")[0][5:]), int(
            latest_run.split("_")[-1].split(".")[0][10:]
        )
        if not c.CONTINUE:
            model_id += 1
            epoch_num = 0
    else:
        model_id, epoch_num = 0, 0

    return model_id, epoch_num


# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
def create_data_slices(cutpoint, c, ds):
    print("==> len ds: ", len(ds))
    print("batch_sizes: ", c.batch_size_train, c.batch_size_val)
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

    print("datasets: ", len(list(train_dataset_slices)), len(list(val_dataset_slices)))
    return train_dataset_slices, val_dataset_slices


# ----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")


# ----------------------------------------------------------------------
