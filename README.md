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

# Development with Dev Container

This project supports development within a Docker container via a Dev Container. This ensures a consistent development environment and avoids the need to install project dependencies directly on your machine.

## Setup

1. Ensure [Docker Desktop](https://www.docker.com/products/docker-desktop/) is installed and running on your system.
2. Ensure that the [Dev Container](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) VS Code extention is installed (ms-vscode-remote.remote-containers)
3. Open the ConnTextUL project in Visual Studio Code.
4. When prompted, choose to reopen the project in a container. (Or Shift+CMD+P and select Dev Containers: Rebuild Container)
5. VS Code (and the Dev Containers extension) will build the container based on the provided Dockerfile and the and devcontainer.json file, run the image on your local machine using Docker Desktop, and then connect your local version of VS Code to the one running in the Docker container. [See here](https://code.visualstudio.com/docs/devcontainers/containers) for details.
6. Once the build is complete, you will have access to a fully configured development environment.

## Using Dev Container

1. All dependencies are pre-installed in the container.
2. Extensions listed in devcontainer.json are automatically available.
3. Use VS Code's terminal and editor to write and execute code in the containerized environment.

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

| Command Line Argument | Argument  | Type  |
|-----------------------|-----------|-------|
| --config              | name of YAML file containing training configuration data | str |

----------------------------------------------------------------------
Configuration options:

| YAML Config Options   | Type | Description | 
|-----------------------|------|-------------|
| device                | str  | specify device to run on 'cpu' or 'gpu' |
| project               | str  | wandb project name |
| num_epochs            | int  | nb of epochs to run |
| batch_size_train      | int  | batch size during training |
| batch_size_val        | int  | batch size during validation |
| num_phon_enc_layers   | int  | number of layers in phonology encoder transformer block |
| num_orth_enc_layers   | int  | number of layers in orthography encoder transformer block |
| num_mixing_enc_layers | int  | number of layers in mixing encoder transformer block |
| num_phon_dec_layers   | int  | number of layers in phonology decoder transformer block |
| num_orth_dec_layers   | int  | number of layers in orthography decoder transformer block |
| learning_rate         | float| learning rate |
| d_model               | int  | Dimensionality of internal model components (embedding, linear and transformer layers). Must be evenly divisble by nhead |
| nhead                 | int  | number of attention heads |
| wandb                 | bool | use Weights & Biases |
| train_test_split      | float| fraction of data in training set. Leave at 1.0 is using test_filenames |
| sweep_filename        | str  | yaml file name for W&B sweep run |
| d_embedding           | int  | Global embedding dimension |
| seed                  | int  | random seed for repeatability |
| model_path            | str  | path to model checkpoint files |
| pathway               | str  | particular pathway to be trained 'o2p', 'p2o', or 'op2op' |
| save_every            | int  | Save data every 'save_every' number of epochs |
| dataset_filename      | str  | The name of the csv file in the data folder that contains the data to train on. (must be in `data/` directory) |
| max_nb_steps          | int  | The maximum number of steps to take per epoch. Good for testing |
| test_filenames        | list | A list of csv files to test the model on once per epoch. (must be in `data/test/` directory) |

----------------------------------------------------------------------

# How to SSH into any of the lab's machine 

Machines in the Computational Science's laboratory likely contain better hardware than what is available on one's own local machine. For that reason, it would
be a good idea to run extensive simulations on them. One might also have the urge to change the code, modify the code base or test minor changes in one of the files.
To perform an ssh jump into any of the machines one requires the following: 

- An FSU ID
- That ID's password

The steps that must be taken for a simple ssh jump are the following:

1. Open any terminal
2. Type the following ```ssh <FSU_ID>@pamd.sc.fsu.edu```
3. You will then be prompted to input your account password, linked to that ID (It is the same password with which you access MyFsu)
4. You will be ssh'ed into the _pamd_ virtual space. From here, you are able to jump into any machine
5. Type the following ```ssh <machine_name>```
6. You will be prompted to type in your password once more. Type it in.

To jump into the machine using VSCODE, and to have any element on it available on VSCode (as long as it is openable):
1. Open the .ssh folder of your machine
2. Create a file named _config_ 
3. In the file named _config_, type in the following:
```
Host pamd
    HostName pamd.sc.fsu.edu
    User <your FSUID>
    Port 22
    
Host spock
    HostName <desired machine>
    User gm23k
    ProxyJump <your FSUID>
    Port 22
```
4. Open VSCode
5. Open the bottom-left remote connections selector
6. Select 'Connect to Host' 
7. Select '+ Add New SSH Host...'
8. Type the following: ```ssh <machine_name>```
9. Select the correct config file that you have just created
10. Close VSCode
11. Open VSCode once more
12. Open the bottom-left remote connections selector
13. Select the machine name that you wish to SSH in, which will now be shown
14. Enter your password however many times it requests it

----------------------------------------------------------------------

2023-07-08
Dependencies: torch, addict
