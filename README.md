# Authors: 
 - Nathan Crock
 - Gordon erlebacher
## Date: 2023-09-18

# Installation with Poetry


- Install poetry through the set of commands for linux :
   - curl -sSL https://install.python-poetry.org | python3 -
   - nano ~/.bashrc   
   - export PATH="/home/name/.local/bin:$PATH"  #add your directory
   - source ~/.bashrc  #applying the changes to the current session
   - poetry --version  #to check if poetry has been installed

- Now once into our project folder , steps use our existing pyproject.toml file to create a poetry shell
   - poetry shell #will create a new venv based on the existing pyproject.toml file 
   - poetry install (or) poetry add wandb@latest 
   - The above commands will install the dependencies required for our project mentioned in the pyproject.toml file
   - Now run the code based on the commands below.

- To create a new poetry project 
   - poetry new project-name
   - cd project-name 
   - Copy and paste the list of dependencies from our original pyproject.toml file to project-name/pyproject.toml
   - poetry shell
   - poetry install
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

| Command Line Argument | Argument  | Type  | Description |
|-----------------------|-----------|-------|-------------|
| --device              | cpu/gpu   | str   | cpu or gpu device |
| --project             | project name  | str   | Project name (no default) |
| --num_epochs          | nb of epochs to run  | int   | Number of epochs |
| --batch_size_train    | batch size during training | int   | Train batch size |
| --batch_size_val      | batch size during validation  | int   | Validation batch size |
| --num_layers          | number of layers | int   | Number of layers |
| --learning_rate       | learning rate | float | Learning rate |
| --continue_training   | N/A       | bool  | Continue training from last checkpoint (default: new run if argument absent) |
| --model_chkpt         | model file name to start from  | str   | Continue training from the checkpoint model_chkpt (assumes --continue_training is present) (default: continue from latest run if --model_id is absent) |
| --d_model             | embedding dimension | int   | Dimensionality of the internal model components including Embedding layer, transformer layers, and linear layers. Must be evenly divisible by nhead |
| --nhead               | number of attention heads | int   | Number of attention heads for all attention modules. Must evenly divided d_model. |
| --wandb               | use Weights & Biases | bool  | Enable wandb (default: disabled if argument absent) |
| --test                | N/A       | bool  | Test mode: only run one epoch on a small subset of the data (default: no test if argument absent) |
| --max_nb_steps        | max number of steps per epoch | int   | Hardcode nb steps per epoch for fast testing. Run full epoch if not present. |
| --train_test_split    | fraction of data in training set  | float | Fraction of data in the training set in [0,1]|
| --which_dataset       | number ot words to read in with --test | int   | Choose the dataset to load |
| --sweep               | yaml file name | str | Yaml file name for W & B sweep run |
| --d_embedding         | Global embedding dimension | int   | Dimensionality of the final embedding layer. |
| --seed                | random seed | int   | Random seed for repeatability. |
| --nb_samples          | number of samples | int   | Number of total samples from dataset. All samples if <=0 |
| --model_path          | path to model checkpoint files  | str   | Path to model checkpoint files. |
| --pathway             | o2p/p2o/op2op | str | Specify the particular pathway to use: o2p, p2o, op2op |
| --save_every          | skip factor for model saves | int   | Save data every 'save_every' number of epochs. Default: 1 |

2023-07-08
Dependencies: torch, addict


# Notes

Regarding running the model with one's GPU, it must be mentioned that the performance will not change by a significant amount. However, it is to noted that the current implementation uses an old and deprecated version of CUDA and pytorch.

Pytorch version = 2.0.1

NVidia Driver Version  = 510.108.03

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.108.03   Driver Version: 510.108.03   CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:0A:00.0  On |                  N/A |
|  0%   41C    P8    23W / 290W |    215MiB /  8192MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      2494      G   /usr/lib/xorg/Xorg                 59MiB |
|    0   N/A  N/A    523242      G   /usr/lib/xorg/Xorg                120MiB |
|    0   N/A  N/A    523385      G   /usr/bin/gnome-shell               16MiB |
+-----------------------------------------------------------------------------+

Current CUDA version = 10.1

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
