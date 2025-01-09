
# BRIDGE

**BRIDGE** is a computational model for naming printed words, designed to address gaps in understanding the cognitive processes underlying reading development. Unlike previous models, BRIDGE maps written (orthographic) and spoken (phonological) representations of potentially unlimited length into a unified global embedding using cross-attention mechanisms. This approach enhances the study of how these modalities combine, particularly for longer words. It also allows selective activation of specific modalities, enabling investigations into how different training methods affect reading development. By bridging cognitive neuroscience and educational technology, BRIDGE offers new insights and tools for research and classroom applications.

---

## Table of Contents
1. [Installation](#installation)  
   - [Install Poetry](#install-poetry)  
   - [Create the Environment](#create-the-environment)
2. [Running the Code](#running-the-code)  
3. [Configuration Files](#configuration-files)  
4. [Development with Dev Container](#development-with-dev-container)    
6. [Authors](#authors)
7. [Dataset](#dataset)

---

## Installation

### Install Poetry

**Poetry** is used to manage dependencies and the Python environment.

#### Linux
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

#### macOS
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
Or via Homebrew:
```bash
brew install poetry
```

#### Windows
> **Note**: Currently, this project is only tested on Unix-like systems (Linux/macOS). Windows support is untested.

#### Validate Installation
Ensure Poetry is installed:
```bash
poetry --version
```
If not, make sure `~/.local/bin` is in your `PATH`:
```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Create the Environment

1. Navigate to the project folder:
   ```bash
   cd BRIDGE
   ```
2. Create and activate a Poetry environment:
   ```bash
   poetry shell
   poetry install
   ```

---

## Running the Code

The BRIDGE training workflow is controlled by configuration files. To launch a training run, execute:
```bash
python app/bin/main.py
```
Parameters such as datasets, model settings, and training configurations are managed through YAML files (see [Configuration Files](#configuration-files)).

---

## Configuration Files

The BRIDGE project uses multiple YAML configuration files located in `BRIDGE/app/config` to handle datasets, models, training, and logging. Below are the details:

### 1. `dataset_config.yaml`

| Parameter                     | Type    | Description                                                                 |
|-------------------------------|---------|-----------------------------------------------------------------------------|
| `dataset_filepath`            | str     | Path to the primary dataset file.                                          |
| `dimension_phon_repr`         | int     | Dimensionality of the phonological representation.                         |
| `orthographic_vocabulary_size`| int     | Size of the orthographic vocabulary.                                       |
| `phonological_vocabulary_size`| int     | Size of the phonological vocabulary.                                       |
| `max_orth_seq_len`            | int     | Maximum sequence length for orthography.                                   |
| `max_phon_seq_len`            | int     | Maximum sequence length for phonology.                                     |

---

### 2. `model_config.yaml`

| Parameter                 | Type   | Description                                                                 |
|---------------------------|--------|-----------------------------------------------------------------------------|
| `d_model`                 | int    | Dimensionality of internal model components (e.g., embeddings, layers). Must be divisible by `nhead`. |
| `d_embedding`             | int    | Dimensionality of global embeddings.                                        |
| `nhead`                   | int    | Number of attention heads in transformer layers.                            |
| `seed`                    | int    | Random seed for reproducibility.                                            |
| `num_phon_enc_layers`     | int    | Number of transformer layers in the phonology encoder.                      |
| `num_orth_enc_layers`     | int    | Number of transformer layers in the orthography encoder.                    |
| `num_mixing_enc_layers`   | int    | Number of transformer layers in the mixing encoder.                         |
| `num_phon_dec_layers`     | int    | Number of transformer layers in the phonology decoder.                      |
| `num_orth_dec_layers`     | int    | Number of transformer layers in the orthography decoder.                    |

---

### 3. `training_config.yaml`

| Parameter              | Type    | Description                                                                 |
|------------------------|---------|-----------------------------------------------------------------------------|
| `num_epochs`           | int     | Number of epochs for training.                                              |
| `batch_size_train`     | int     | Batch size for training.                                                    |
| `batch_size_val`       | int     | Batch size for validation.                                                  |
| `train_test_split`     | float   | Fraction of data used for training.                                         |
| `learning_rate`        | float   | Learning rate for the optimizer.                                            |
| `training_pathway`     | str     | Training pathway to use (`p2p`, `o2p`, etc.).                               |
| `device`               | str     | Device to run training on (`'cpu'` or `'gpu'`).                             |
| `save_every`           | int     | Save model checkpoints every N epochs.                                      |
| `model_artifacts_dir`  | str     | Directory to save model artifacts.                                          |
| `weight_decay`         | float   | Weight decay parameter for regularization.                                  |

---

### 4. `sweep_config.yaml`

| Parameter         | Type    | Description                                                                 |
|-------------------|---------|-----------------------------------------------------------------------------|
| `program`         | str     | Path to the sweep script.                                                   |
| `method`          | str     | Sweep method (`bayes`, `grid`, or `random`).                                |
| `name`            | str     | Name of the sweep.                                                          |
| `metric`          | dict    | Sweep optimization metric with `goal` (e.g., `minimize`) and `name`.        |
| `parameters`      | dict    | Parameter grid for the sweep, including batch sizes, learning rates, etc.   |

---

### 5. `wandb_config.yaml`

| Parameter         | Type    | Description                                                                 |
|-------------------|---------|-----------------------------------------------------------------------------|
| `project`         | str     | Name of the W&B project for logging.                                        |
| `entity`          | str     | W&B entity or team name.                                                    |
| `is_enabled`      | bool    | Whether W&B logging is enabled.                                             |

---

## Development with Dev Container

This repository supports development using [Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers) in Visual Studio Code, which ensures a consistent and reproducible environment for development.

### Features of the Dev Container Setup

1. **Dynamic Platform Detection**:
   - The script `detect-platform.sh` dynamically determines whether a CPU-based or GPU-based Dockerfile should be used (`Dockerfile.cpu.dev` or `Dockerfile.gpu.dev`).
   - The selected platform configuration is saved in `.devcontainer/platform-config.json`.

2. **Custom VS Code Configuration**:
   - Includes Python-specific settings, such as:
     - Default Python interpreter set to `/usr/local/bin/python`.
     - Auto-discovery of `pytest` for testing.
     - Auto-formatting with `Black`.
   - Extensions pre-installed in the container:
     - Python (e.g., `ms-python.python`, `ms-python.black-formatter`, `ms-toolsai.jupyter`).
     - Docker (e.g., `ms-azuretools.vscode-docker`).
     - GitHub (e.g., `GitHub.copilot`, `GitHub.pull-request-github`).
     - Jupyter Notebook support.

3. **Container User and Environment Variables**:
   - The container runs as the root user.
   - Includes `PYTHONPATH` set to the container's workspace folder.

4. **Testing Framework**:
   - Configured to use `pytest` with arguments:
     - Tests are discovered in the `tests/` folder.
     - Run in verbose mode with detailed output.

### Setup Instructions

1. **Install Required Tools**:
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop) and ensure it is running.
   - Install the [Dev Containers VS Code Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

2. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd BRIDGE
   ```

3. **Open in VS Code**:
   - Open the project folder in VS Code.
   - Reopen the project in a Dev Container:
     - Either accept the prompt to reopen in a container.
     - Or manually select "Reopen in Container" from the Command Palette (`Shift + Cmd + P` or `Ctrl + Shift + P`).

4. **Build and Initialize**:
   - During initialization, the container detects the platform (CPU/GPU) and configures the environment accordingly.
   - You will see logs confirming the setup:
     - Example: `Container created for platform: ...`.

5. **Development in the Container**:
   - Use the integrated terminal to run commands like:
     ```bash
     poetry install
     poetry shell
     pytest tests
     ```

6. **Custom Dockerfile**:
   - CPU Development: Dockerfile.cpu.dev.
   - GPU Development: Dockerfile.gpu.dev.

### Using Dev Container

- All dependencies are pre-installed in the container.  
- Use VS Code’s integrated terminal to run commands (`poetry shell`, etc.).  
---

## Dataset
<!-- TODO -->
---
