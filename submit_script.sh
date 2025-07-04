#!/bin/bash

# A more robust script for submitting SLURM jobs.

# Check if the first argument is a dependency flag.
if [[ $1 == --dependency=* ]]; then
    DEPENDENCY_ARG="$1"
    shift  # Remove the dependency argument from the list.
else
    DEPENDENCY_ARG=""
fi

# The next argument must be the SLURM script file.
SLURM_FILE=$1
shift # Remove the SLURM file from the list.

# All remaining arguments are for the Python script inside SLURM.
# Store them in an array to preserve spaces and special characters.
SCRIPT_AND_ARGS=("$@")

# Use --parsable to get only the job ID, making output clean and reliable.
if [ -n "$DEPENDENCY_ARG" ]; then
    echo "Submitting with dependency: $DEPENDENCY_ARG"
    sbatch --parsable "$DEPENDENCY_ARG" "$SLURM_FILE" "${SCRIPT_AND_ARGS[@]}"
else
    echo "Submitting without dependency"
    sbatch --parsable "$SLURM_FILE" "${SCRIPT_AND_ARGS[@]}"
fi