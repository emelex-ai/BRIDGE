#!/bin/bash -x
#SBATCH --job-name=apptainer_python_job
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=75GB
#SBATCH --time=01:00:00
#SBATCH -A pilotgpu

echo "Inside slurm script"

# --- 1. Get Python script and its arguments from command-line ---
if [ -z "$1" ]; then
    echo "Error: No Python script provided."
    echo "Usage: sbatch $0 your_script.py [args...]"
    exit 1
fi
PYTHON_SCRIPT=$1
shift  # Shift off the script name, the rest are the arguments for it.
SCRIPT_ARGS=("$@") # Store arguments in an array

# --- 2. Set up cache directories in scratch space ---
SCRATCH_CACHE="${TMPDIR:-/tmp}/cache_${SLURM_JOB_ID}"
export UV_CACHE_DIR="$SCRATCH_CACHE/uv"
export PIP_CACHE_DIR="$SCRATCH_CACHE/pip"
export HF_HOME="$SCRATCH_CACHE/huggingface"
export XDG_CACHE_HOME="$SCRATCH_CACHE"

# --- 3. Set up paths using SLURM_SUBMIT_DIR ---
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
echo "Project root: ${PROJECT_ROOT}"
SHARED_VENV="$PROJECT_ROOT/.venv"
CONTAINER_IMAGE="$HOME/containers/cuda_uv_12.sif"

echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Project root: $PROJECT_ROOT"
echo "Shared environment: $SHARED_VENV"
echo "Container image: $CONTAINER_IMAGE"

# --- 4. Run the container with shared environment ---
echo "Running: $PYTHON_SCRIPT with args: ${SCRIPT_ARGS[*]}"

# Use bash -c '...' and pass arguments to it to avoid complex quoting issues.
# The first argument after the script string is $0 for the inner shell.
apptainer exec \
    --nv \
    --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
    --bind "$SHARED_VENV:$SHARED_VENV" \
    --bind "${TMPDIR:-/tmp}:${TMPDIR:-/tmp}" \
    "$CONTAINER_IMAGE" \
    bash -c '
        echo "--- Inside container ---"
        set -x # Enable debugging inside container
        cd "$1"
        source "$2/bin/activate"
        python -u "$3" "${@:4}"
    ' _bash "$SLURM_SUBMIT_DIR" "$SHARED_VENV" "$PYTHON_SCRIPT" "${SCRIPT_ARGS[@]}"
