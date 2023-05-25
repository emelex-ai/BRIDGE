from dataset import ConnTextULDataset
from model import Model
import torch as pt
import wandb
import tqdm
import sys
import math
import glob

# I have to first type this line from the command line and enter the API key
MODEL_PATH = "./models"
ds = ConnTextULDataset()
wandb.login()


if pt.cuda.is_available():
    device = pt.device("cuda:0")
else:
    device = pt.device("cpu")

# device = 'cpu'  # GE commented it out
print("device = ", device)

exp_count = 1
num_epochs = 100
CONTINUE = False
learning_rate = 1e-3
batch_size = 32
train_test_split = 0.8
cutpoint = int(train_test_split * len(ds))
d_model = 128
nhead = 4
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
model_runs = glob.glob(MODEL_PATH + "/model*")  # added model to the glob
print("model_runs: ", model_runs)
if model_runs:
    # GE comments
    # Whatever numbering you are using for model_runs, you should use integers with leading zeros, or else sorted will not work correctly.
    # Sorting on letters is dangerous since different people might have different sorting conventions
    latest_run = sorted(model_runs)[-1].split("/")[-1]
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

# Launch `exp_count` experiments, trying different dropout rates
for _ in range(exp_count):
    if CONTINUE:
        chkpt = pt.load(MODEL_PATH + f"/model{model_id}_checkpoint{epoch_num}.pth")
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
        )
        # Is AdamW always the best? I do know that Adam with weight_decay is broken
        opt = pt.optim.AdamW(model.parameters(), learning_rate)

    # 🐝 initialise a wandb run
    run = wandb.init(
        entity="emelex",  # GE: Necessary because I am in multiple teams
        project="GE_ConnTextUL_WandB",
        config={
            "starting_epoch": epoch_num,
            "epochs": num_epochs,
            "batch_size": batch_size,
            "d_model": d_model,  
            "nhead": nhead,
            "lr": learning_rate,
            "id": model_id,
        },
    )

    generated_text_table = wandb.Table(columns=["Step", "Generated Output"])

    # Copy your config
    config = run.config

    pbar = tqdm.tqdm(range(epoch_num, epoch_num + num_epochs), position=0)

    # Wandb: Add the model to be logged. By default, the data is logged every 1000 steps. 
    # log="all": log gradients and parameters
    # To output every epoch, use run.log() with specific parameters
    # If using wandb.log, one must calculate our own gradients and parameters
    run.watch(model, log="all")

    model.to(device)

    # Training
    example_ct = 0
    for epoch in pbar:
        model.train()
        print("\nTraining Loop...")
        for step, batch_slice in enumerate(train_dataset_slices):
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

            # NEW NEW NEW NEW NEW NEW NEW NEW NEW

            # Accuracy function for orthography.
            # Take argmax of orthographic logits for accuracy comparison:
            A_orth = pt.argmax(logits["orth"], dim=1)
            # Keep orthographic encoder input ids unchanged:
            B_orth = orthography["enc_input_ids"][:, 1:]
            # Compute orthographic accuracy:
            orth_accuracy = (
                pt.tensor(pt.where((A_orth == B_orth).all(dim=1))[0].size())
                / pt.tensor(A_orth.size())[0]
            )

            # Accuracy function for phonology.
            # Compute dimensions for phonological logit and target reshaping:
            oldshape = logits["phon"].size()
            newshape_0 = oldshape[0] * oldshape[2]
            newshape_1 = oldshape[3]
            # Take argmax of phonological logits for accuracy comparison, reshaping to get a square tensor:
            A_phon = pt.argmax(logits["phon"], dim=1)
            A_phon = pt.reshape(A_phon, [newshape_0, newshape_1])
            print("\nA_phon shape:", A_phon.size())
            # Reshape phonological targets:
            B_phon = phonology["targets"]
            B_phon = pt.reshape(B_phon, [newshape_0, newshape_1])
            print("B_phon shape:", B_phon.size(), "\n")
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
                generation = model.generate(
                    orthography, orthography_mask, phonology, phonology_mask
                )
                generated_text = ds.character_tokenizer.decode(
                    generation["orth"].tolist()
                )[0]
                # Log the text in the WandB table. GE: should it be every step or every epoch? 
                generated_text_table.add_data(step, generated_text)

            example_ct += len(batch["orthography"])
            metrics = {
                "train/orth_loss": orth_loss,
                "train/phon_loss": phon_loss,
                "train/train_loss": loss,
                "train/epoch": epoch,
                "train/example_ct": example_ct,
                "train/global_embedding_magnitude": pt.norm(
                    model.global_embedding[0], p=2
                ),
                "train/model_weights_magnitude": pt.sqrt(
                    sum([pt.norm(w[0], p=2) ** 2 for w in model.parameters()])
                ),
                # "train/lr": opt.state_dict()['param_groups'][0]['lr'],
                "train/generated_text_table": generated_text_table,
                "orthographic accuracy": orth_accuracy,
                "phonological accuracy": phon_accuracy,
            }

        # GE: Log the metrics in the wandb run every epoch
        run.log(metrics)  # GE: added

        model.eval()
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
                run.log(more_metrics)  # GE: How often do you want this output? Currently every step on prediction

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
