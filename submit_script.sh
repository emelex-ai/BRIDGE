#!/bin/bash -x

#SCRIPT=$1
#PYTHON=$2
#sbatch $1 $2

# Example: 
#
# bash submit_script.sh run_in_apptainer.slurm my_script.py

# #!/bin/bash
# Modified submit_slurm.sh

# Check if first argument starts with --dependency
if [[ $1 == --dependency=* ]]; then
    DEPENDENCY_ARG="$1"
    shift  # Remove dependency argument
else
    DEPENDENCY_ARG=""
fi

SLURM_FILE=$1
SCRIPT_TO_RUN=$2
shift 2
SCRIPT_ARGS="$@"

if [ -n "$DEPENDENCY_ARG" ]; then
    echo "Submitting with dependency: $DEPENDENCY_ARG"
    sbatch "$DEPENDENCY_ARG" "$SLURM_FILE" "$SCRIPT_TO_RUN" $SCRIPT_ARGS
else
    echo "Submitting without dependency"
    sbatch "$SLURM_FILE" "$SCRIPT_TO_RUN" $SCRIPT_ARGS
fi