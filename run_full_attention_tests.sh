#!/bin/bash

# Configuration
SEQ_LENS=(512 1024 2048 4096 8192)
D_MODEL=512
NHEAD=8
BATCH_SIZE=4
OUTPUT_CSV="full_attention_results.csv"

# Remove existing CSV file
rm -f $OUTPUT_CSV

echo "Starting full attention tests..."
echo "Results will be saved to: $OUTPUT_CSV"

# Submit jobs sequentially with dependencies
PREV_JOB=""

for seq_len in "${SEQ_LENS[@]}"; do
    echo "Submitting job for seq_len=$seq_len"

    if [ -z "$PREV_JOB" ]; then
        # First job - no dependency
        JOB_ID=$(./submit_script.sh script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE "$OUTPUT_CSV")
    else
        # Subsequent jobs - depend on previous job. No more grep!
        JOB_ID=$(./submit_script.sh --dependency=afterok:$PREV_JOB script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE "$OUTPUT_CSV")
    fi

    # Check if a valid job ID was returned.
    if [[ -n "$JOB_ID" && "$JOB_ID" -gt 0 ]]; then
        echo "  Submitted job $JOB_ID"
        PREV_JOB=$JOB_ID
    else
        echo "  ERROR: Failed to submit job for seq_len=$seq_len. Aborting."
        # Optional: Cancel previously submitted jobs in the chain.
        if [ -n "$PREV_JOB" ]; then
            scancel "$PREV_JOB"
            echo "  Canceled job chain."
        fi
        exit 1
    fi
done

echo ""
echo "All full attention jobs submitted!"
echo "Final job ID: $PREV_JOB"
echo "Monitor progress with: squeue -u \$USER"
echo "Check results with: cat $OUTPUT_CSV"
