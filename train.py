from dataset import ConnTextULDataset
from model import Model
import torch
import tqdm
import sys
import math
import glob
import argparse
import wandb

parser = argparse.ArgumentParser(description='Train a ConnTextUL model')

parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--num_layers", type=int, default=4, help="Number of layers")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--continue_training", type=bool, default=False, help="Continue training from last checkpoint")
parser.add_argument("--d_model", type=int, default=128, help="Dimensionality of the internal model components \
                                                              including Embedding layer, transformer layers, \
                                                              and linear layers. Must be evenly divisible by nhead")
parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads for all attention modules. \
                                                          Must evenly divide d_model.")
parser.add_argument("--test", type=bool, default=False, help="Test mode: only run one epoch on a small subset of the data")

args = parser.parse_args()

num_epochs = args.num_epochs
batch_size = args.batch_size
num_layers = args.num_layers
learning_rate = args.learning_rate
CONTINUE = args.continue_training
TEST = args.test
nhead = args.nhead
d_model = args.d_model
assert d_model%nhead == 0, "d_model must be evenly divisible by nhead"

MODEL_PATH = './models'

if TEST:
  ds = ConnTextULDataset(test=True)
  d_model = 16
  nhead = 2
  num_layers = 2
  batch_size = 1
  learning_rate = 0.001
  num_epochs = 1
  CONTINUE = False
  seed = 1337
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.use_deterministic_algorithms(True)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
else:
  ds = ConnTextULDataset()

if torch.cuda.is_available():
  device = torch.device('cuda:0')
else:
  device = torch.device('cpu')

device = 'cpu'

train_test_split = 0.95
cutpoint = int(train_test_split * len(ds))
train_dataset_slices = []
for batch in range(math.ceil(cutpoint/batch_size)):
  train_dataset_slices.append(slice(batch*batch_size, min((batch+1)*batch_size, cutpoint)))

val_dataset_slices = []
for batch in range(math.ceil((len(ds)-cutpoint)/batch_size)):
  val_dataset_slices.append(slice(cutpoint+batch*batch_size, min(cutpoint+(batch+1)*batch_size, len(ds))))

# Get latest model run information
model_runs = glob.glob(MODEL_PATH+'/*')
if model_runs:
  latest_run = sorted(model_runs)[-1].split('/')[-1]
  model_id, epoch_num = int(latest_run.split("_")[0][5:]), int(latest_run.split("_")[-1].split('.')[0][10:])
  if not CONTINUE:
    model_id += 1
    epoch_num = 0
else:
  model_id, epoch_num = 0, 0

# A number for WandB:
n_steps_per_epoch = len(train_dataset_slices)

if CONTINUE:
  chkpt = torch.load(MODEL_PATH+f"/model{model_id}_checkpoint{epoch_num}.pth")
  model = Model(orth_vocab_size=len(ds.character_tokenizer), 
                phon_vocab_size=len(ds.phonology_tokenizer), 
                d_model=chkpt['d_model'], 
                nhead=chkpt['nhead'],
                num_layers=2)
  model.load_state_dict(chkpt['model'])
  opt = torch.optim.AdamW(model.parameters(), learning_rate)
  opt.load_state_dict(chkpt['optimizer'])
else:
  model = Model(len(ds.character_tokenizer), len(ds.phonology_tokenizer), d_model=d_model, nhead=nhead)
  opt = torch.optim.AdamW(model.parameters(), learning_rate)

# 🐝 initialise a wandb run
wandb.login()
run = wandb.init(
          project="ConnTextUL_WandB",
          config={
              "starting_epoch": epoch_num,
              "epochs": num_epochs,
              "batch_size": batch_size,
              'd_model': d_model,
              'nhead': nhead,
              "lr": learning_rate,
              "id": model_id
              })

generated_text_table = wandb.Table(columns=["Step", "Generated Output"])

# Copy your config 
config = run.config

pbar = tqdm.tqdm(range(epoch_num, epoch_num+num_epochs), position=0)

run.watch(model, log='all')

model.to(device)

#Training
example_ct = 0
for epoch in pbar:
    model.train()
    print("\nTraining Loop...")
    for step, batch_slice in enumerate(train_dataset_slices):
        batch = ds[batch_slice]
        orthography, phonology = batch['orthography'].to(device), batch['phonology'].to(device)
        logits = model(orthography['enc_input_ids'], orthography['enc_pad_mask'],
                    orthography['dec_input_ids'], orthography['dec_pad_mask'],
                    phonology['enc_input_ids'], phonology['enc_pad_mask'],
                    phonology['dec_input_ids'], phonology['dec_pad_mask'])
                  
        orth_loss = torch.nn.CrossEntropyLoss(ignore_index=4)(logits['orth'], orthography['enc_input_ids'][:,1:]) 
        phon_loss = torch.nn.CrossEntropyLoss(ignore_index=2)(logits['phon'], phonology['targets'])
        loss = orth_loss + phon_loss

        loss.backward()
        opt.step()
        opt.zero_grad()



        #NEW NEW NEW NEW NEW NEW NEW NEW NEW

        # Accuracy function for orthography.
        # Take argmax of orthographic logits for accuracy comparison:
        A_orth = torch.argmax(logits['orth'], dim=1)
        # Keep orthographic encoder input ids unchanged:
        B_orth = orthography['enc_input_ids'][:,1:]
        # Compute orthographic accuracy:
        word_accuracy = torch.tensor(torch.where((A_orth == B_orth).all(dim=1))[0].size())/torch.tensor(A_orth.size())[0]
        char_accuracy = (A_orth == B_orth).sum()/torch.tensor(A_orth.shape).prod()

        # Accuracy function for phonology.
        # Compute dimensions for phonological logit and target reshaping:
        oldshape = logits['phon'].size()
        newshape_0 = oldshape[0] * oldshape[2]
        newshape_1 = oldshape[3]
        # Take argmax of phonological logits for accuracy comparison, reshaping to get a square tensor:
        A_phon = torch.argmax(logits['phon'], dim=1)
        A_phon = torch.reshape(A_phon, [newshape_0, newshape_1])
        #print('\nA_phon shape:', A_phon.size())
        # Reshape phonological targets:
        B_phon = phonology['targets']
        B_phon = torch.reshape(B_phon, [newshape_0, newshape_1])
        #print('B_phon shape:', B_phon.size(),'\n')
        # Compute phonoloigcal accuracy:
        phon_accuracy = torch.tensor(torch.where((A_phon == B_phon).all(dim=1))[0].size())/torch.tensor(A_phon.size())[0]

        #END END END END END END END END END



        # Now we generate orthographic tokens and phonological vectors for the input 'elephant'
        if 1:
          orth = ds.character_tokenizer.encode(['taking'])
          orthography = orth['enc_input_ids'].to(device)
          orthography_mask = orth['enc_pad_mask'].to(device)
          phon = ds.phonology_tokenizer.encode(['taking'])
          phonology = [[t.to(device) for t in tokens] for tokens in phon['enc_input_ids']]
          phonology_mask = phon['enc_pad_mask'].to(device)
          generation = model.generate(orthography, orthography_mask, phonology, phonology_mask)
          generated_text = ds.character_tokenizer.decode(generation['orth'].tolist())[0]
          # Log the text in the WandB table
          generated_text_table.add_data(step, generated_text)

        example_ct += len(batch['orthography'])
        metrics = {"train/orth_loss": orth_loss,
                    "train/phon_loss": phon_loss,
                    "train/train_loss": loss, 
                    "train/epoch": epoch, 
                    "train/example_ct": example_ct,
                    "train/global_embedding_magnitude": torch.norm(model.global_embedding[0], p=2),
                    "train/model_weights_magnitude": torch.sqrt(sum([torch.norm(w[0],p=2)**2 for w in model.parameters()])),
                    #"train/lr": opt.state_dict()['param_groups'][0]['lr'],
                    "train/generated_text_table": generated_text_table,
                    "word accuracy": word_accuracy,
                    "phonological accuracy": phon_accuracy,
                    "character accuracy": char_accuracy,
                  }
        run.log(metrics)

    model.eval()
    with torch.no_grad():
      print("\nValidation Loop...")
      for step, batch_slice in enumerate(val_dataset_slices):
          batch = ds[batch_slice]
          orthography, phonology = batch['orthography'].to(device), batch['phonology'].to(device)
          logits = model(orthography['enc_input_ids'], orthography['enc_pad_mask'],
                      orthography['dec_input_ids'], orthography['dec_pad_mask'],
                      phonology['enc_input_ids'], phonology['enc_pad_mask'],
                      phonology['dec_input_ids'], phonology['dec_pad_mask'])
        
          val_loss = torch.nn.CrossEntropyLoss(ignore_index=4)(logits['orth'] , orthography['enc_input_ids'][:,1:]) 
          val_loss = val_loss + torch.nn.CrossEntropyLoss(ignore_index=2)(logits['phon'], phonology['targets'])              
          more_metrics = {"val/val_loss": val_loss}             
          run.log(more_metrics)
    
    torch.save({
      'epoch': epoch,
      'batch_size': batch_size,
      'd_model': d_model,
      'nhead': nhead,
      'model': model.state_dict(),
      'optimizer': opt.state_dict()
    },
    MODEL_PATH + f"/model{model_id}_checkpoint{epoch}.pth")

if TEST:
  with open('test_results.txt', 'w') as f:
    f.write(f"val loss: {val_loss}\n")
    f.write(f"word accuracy: {word_accuracy}\n")
    f.write(f"phonological accuracy: {phon_accuracy}\n")
    f.write(f"character accuracy: {char_accuracy}\n")
    f.write(f"generated text: {generated_text}\n")

# 🐝 Close your wandb run 
run.finish()