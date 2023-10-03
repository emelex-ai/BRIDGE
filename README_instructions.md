# Author: G. erlebacher
## Date: 2023-09-18

# Installation with Poetry
- Clone from github

  github clone https://github.com/erlebach/ConnTextUL_poetry.git

- Create the virtual environment (./.venv)

   poetry install

- Enter the Poetry shell

   poetry shell

- Run one of the scripts below

Github: ConnTextUL_poetry

# Running the code

How to run the code. I provide four example files, stored in scripts/: 

run_test.x		
run_wandb.x		
run_wandb_test.x


1. make sure "--project project_name" is included in the arguments

  Example:   
       python -m src.main --project "nathan"

----------------------------------------------------------------------
2. To use wandb, add --wandb

       python -m src.main --project "nathan"  \
	                      --wandb
----------------------------------------------------------------------
3. To run in test mode, 
       python -m src.main --project "nathan"  \
	                      --wandb \
						  --test
----------------------------------------------------------------------
4. To run in sweep mode, 
       python -m src.main --project "nathan"  \
	                      --wandb \
						  --test
						  --sweep "nathan_sweep.yaml"
----------------------------------------------------------------------
To run a sweep, simply add 

   --sweep "sweep_file.yaml"

where "sweep_file.yaml" is a yaml file that contains the sweep parameters. 
----------------------------------------------------------------------
# ISSUES: 
- I cannot see metric graphs on wandb. I cannot figure out the error. Possibly it 
is related to something in the WandbWrapper class. 

- Sweep, wandb with no sweep, and no wandb are all working. 
----------------------------------------------------------------------
# Unit Tests

2023-10-01
Test new run with d_model=128 and d_model=64  (output files shoudl be different). Run in test mode 5 epochs.

----------------------------------------------------------------------
Program arguments: 
|-----------------------|-----------|-------|-------------|
| Command Line Argument | Argument  | Type  | Description |
|-----------------------|-----------|-------|-------------|
| --device              | cpu/gpu   | str   | cpu or gpu device |
| --project             | <name>    | str   | Project name (no default) |
| --num_epochs          | <int>     | int   | Number of epochs |
| --batch_size_train    | <int>     | int   | Train batch size |
| --batch_size_val      | <int>     | int   | Validation batch size |
| --num_layers          | <int>     | int   | Number of layers |
| --learning_rate       | <float>   | float | Learning rate |
| --continue_training   | N/A       | bool  | Continue training from last checkpoint (default: new run if argument absent) |
| --model_chkpt         | <path>    | str   | Continue training from the checkpoint model_chkpt (assumes --continue_training is present) (default: continue from latest run if --model_id is absent) |
| --d_model             | <int>     | int   | Dimensionality of the internal model components including Embedding layer, transformer layers, and linear layers. Must be evenly divisible by nhead |
| --nhead               | <int>     | int   | Number of attention heads for all attention modules. Must evenly divided d_model. |
| --wandb               | N/A       | bool  | Enable wandb (default: disabled if argument absent) |
| --test                | N/A       | bool  | Test mode: only run one epoch on a small subset of the data (default: no test if argument absent) |
| --max_nb_steps        | <int>     | int   | Hardcode nb steps per epoch for fast testing |
| --train_test_split    | <float>   | float | Fraction of data in the training set |
| --which_dataset       | <int>     | int   | Choose the dataset to load |
| --sweep               | <str>     | str   | Run a sweep from a configuration file |
| --d_embedding         | <int>     | int   | Dimensionality of the final embedding layer. |
| --seed                | <int>     | int   | Random seed for repeatability. |
| --nb_samples          | <int>     | int   | Number of total samples from dataset. All samples if <=0 |
| --model_path          | <path>    | str   | Path to model checkpoint files. |
| --pathway             | o2p/p2o/op2op | str | Specify the particular pathway to use: o2p, p2o, op2op |
| --save_every          | <int>     | int   | Save data every 'save_every' number of epochs. Default: 1 |

