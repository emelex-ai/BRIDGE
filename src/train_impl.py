from src.wandb_wrapper import WandbWrapper, MyRun
from torch.utils.data import Dataset
from src.model import Model
import torch as pt
import time
import math
import glob

# WandbWrapper is a singleton
wandb = WandbWrapper()
run = MyRun()


def evaluate_model(model, val_dataset_slices, device, opt, ds, mode):
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
            more_metrics = {mode+"/loss": orth_loss + phon_loss,
                            mode+"/orth_loss": orth_loss,
                            mode+"/phon_loss": phon_loss,}

    return more_metrics

#----------------------------------------------------------------------
def single_step(c, pbar, model, train_dataset_slices, batch_slice, ds, device, opt, epoch, step, generated_text_table, example_ct, mode):
    """ """
    batch = ds[batch_slice]
    orthography = batch["orthography"].to(device)
    phonology   = batch["phonology"].to(device)
    # In an ideal world, a DataLoader should be used: 
    # then the call to the model could be: model(dataloader.next())  (pseudocode)
    # GE: I do not understand why one inputs decoder input ids into the model
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
    metrics = compute_metrics(logits, orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table, mode)
    return metrics

#--------------------------------------------------------------------
def calculate_accuracies(logits, orthography, phonology):

    # --- Calculate Orthographic Accuracy ---
    # Determine model predictions by taking argmax of orthographic logits
    orth_pred = pt.argmax(logits["orth"], dim=1)
    # Use the orthographic encoder input ids as labels
    # notice we slice from 1: onwards to remove the start token (not predicted)
    orth_true = orthography['enc_input_ids'][:,1:]

    # Create a mask for valid positions (not padding)
    orth_valid_mask = (orth_true != 4)

    # Apply the mask to true and predicted values
    masked_orth_true = orth_true[orth_valid_mask]
    masked_orth_pred = orth_pred[orth_valid_mask]

    # Calculate letter-wise accuracy
    correct_matches = (masked_orth_pred == masked_orth_true).sum()
    letter_wise_accuracy = correct_matches.float() / orth_valid_mask.sum().float()

    # To calculate word-wise accuracy, we need to check if all letters in a single word are correct
    orth_pred[~orth_valid_mask] = 4
    word_wise_mask = (orth_pred == orth_true)
    orth_word_accuracy = word_wise_mask.all(dim=1).float().mean()

    # --- Calculate Phonological Accuracy ---
    phon_pred = pt.argmax(logits["phon"], dim=1)
    phon_true = phonology['targets']

    # Create a mask for valid positions (not padding)
    phon_valid_mask = (phon_true != 2)

    # Apply the mask to true and predicted values
    masked_phon_true = phon_true[phon_valid_mask]
    masked_phon_pred = phon_pred[phon_valid_mask]

    # Phoneme segment accuracy
    correct_phoneme_segments = (masked_phon_pred == masked_phon_true).sum()
    phon_segment_accuracy = correct_phoneme_segments.float() / phon_valid_mask.sum().float()

    # Phoneme-wise accuracy
    phoneme_wise_mask = (phon_pred == phon_true)
    phoneme_wise_accuracy = phoneme_wise_mask.all(dim=-1).sum()/(masked_phon_true.shape[0]/phon_true.shape[-1])

    # Phoneme word accuracy
    word_accuracies = [word[target != 2].all().int() for word, target in zip(phoneme_wise_mask, phon_true)]
    #phon_word_accuracy = phoneme_wise_mask.all(dim=-1).all(dim=-1).sum()/phon_true.shape[0]
    phon_word_accuracy = sum(word_accuracies)/len(word_accuracies)

    output = {"letter_wise_accuracy": letter_wise_accuracy,
              "word_wise_accuracy": orth_word_accuracy,
              "phon_segment_accuracy": phon_segment_accuracy,
              "phoneme_wise_accuracy": phoneme_wise_accuracy,
              "phon_word_accuracy": phon_word_accuracy}
    
    return output


#----------------------------------------------------------------------
def generate(ds, device):
    """
    Generative model: Given the first character, generate the following ones 
    Question: do we also feed the first phoneme?
    """
    orth = ds.character_tokenizer.encode(["elephant"])
    print("orth: ", orth)
    orthography = orth["enc_input_ids"].to(device)
    print("orthography: ", orthography)
    orthography_mask = orth["enc_pad_mask"].to(device)
    phon = ds.phonology_tokenizer.encode(["elephant"])  # None. WHY? 
    print("phon: ", phon)

    #"""
    print("device= ", device)
    print("phon: ", phon["enc_input_ids"])
    for tokens in phon["enc_input_ids"]:
        print("tokens: ", tokens)
        for t in tokens:
            print("  t: ", t)
            t.to(device)
    #"""


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

#----------------------------------------------------------------------
def compute_metrics(logits,orthography, phonology, batch, example_ct, orth_loss, phon_loss, loss, epoch, step, ds, device, model, generated_text_table, mode):

    example_ct[0] += len(batch["orthography"])
    metrics = {
        mode+"/orth_loss": orth_loss,
        mode+"/phon_loss": phon_loss,
        mode+"/train_loss": loss,
        mode+"/epoch": epoch,
        mode+"/example_ct": example_ct[0],  # GE: put in a list so I could use it in a function argument
        mode+"/global_embedding_magnitude": pt.norm(
            model.global_embedding[0], p=2
        ),
        mode+"/model_weights_magnitude": pt.sqrt(
            sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
        ),
        # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
        mode+"/generated_text_table": generated_text_table,
    }

    accuracies = calculate_accuracies(logits, orthography, phonology)
    for accuracy in accuracies:
        metrics[mode+"/"+accuracy] = accuracies[accuracy]

    return metrics
#----------------------------------------------------------------------
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

    print("==> train_single_epoch: nb steps: ", len(list(dataset_slices)))

    metrics = [{}]
    for step, batch_slice in enumerate(dataset_slices):
        if c.max_nb_steps > 0 and nb_steps >= c.max_nb_steps:
            break
        metrics[0] = single_step_fct(batch_slice, step, epoch, "train") 
        nb_steps += 1
        print("train: nb_steps: ", nb_steps)

    metrics = metrics[0]
    metrics['time_per_train_step'] = (time.time() - start) / nb_steps
    metrics['time_per_train_epoch'] = c.n_steps_per_epoch * metrics['time_per_train_step']
    return metrics
#----------------------------------------------------------------------
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
    metrics['time_per_val_step'] = (time.time() - start) / nb_steps
    metrics['time_per_val_epoch'] = c.n_steps_per_epoch * metrics['time_per_val_step']
    return metrics

#----------------------------------------------------------------------
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
        if not c.continue_training:
            model_id += 1
            epoch_num = 0
    else:
        model_id, epoch_num = 0, 0

    return model_id, epoch_num
#----------------------------------------------------------------------
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
    print("==> len ds: ", len(ds))
    print("batch_sizes: ", c.batch_size_train, c.batch_size_val)
    train_dataset_slices = []
    for batch in range(math.ceil(cutpoint / c.batch_size_train)):
        train_dataset_slices.append(
            slice(batch * c.batch_size_train, min((batch + 1) * c.batch_size_train, cutpoint))
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

#----------------------------------------------------------------------
def print_weight_norms(model, msg):
    norm = pt.sqrt(sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()]))
    print(f"==> {msg}, {norm}")

#----------------------------------------------------------------------
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
    #wandb.log({"hist_embeddings": [wandb.Histogram(orth_embed_weights), wandb.Histogram(phon_embed_weights)]})
    # log the embeddings as two tables: one for orthography, one for phonology

    def fill_table(A):
        n, m = A.numpy().shape
        columns = ['col'+str(i) for i in range(m)]
        return wandb.Table(data=A.numpy().tolist(), columns=columns)

    orth_embed_table = fill_table(orth_embed_weights)
    phon_embed_table = fill_table(orth_embed_weights)

    wandb.log({"orth_embed_table": orth_embed_table, "phon_embed_table": phon_embed_table})

#----------------------------------------------------------------------
