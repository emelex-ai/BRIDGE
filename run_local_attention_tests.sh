#!/bin/bash

# Configuration
SEQ_LENS=(512 1024 2048 4096 8192)
D_MODEL=512
NHEAD=8
BATCH_SIZE=4
WINDOW_SIZES=(64 128 256)
OUTPUT_CSV="local_atention_results.csv"

# Remove existing CSV file
rm -f $OUTPUT_CSV

echo "Starting sliding window tests..."
echo "Results will be saved to: $OUTPUT_CSV"

# Submit jobs sequentially with dependencies
PREV_JOB=""

for seq_len in "${SEQ_LENS[@]}"; do
  for ws in "${WINDOW_SIZES[@]}"; do
    echo "Submitting job for seq_len=$seq_len, window_size=$ws"

    if [ -z "$PREV_JOB" ]; then
        # First job - no dependency
        JOB_ID=$(./submit_script.sh script.slurm bridge/domain/model/test_single_local_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE $ws "$OUTPUT_CSV")
    else
        # Subsequent jobs - depend on previous job.
        JOB_ID=$(./submit_script.sh --dependency=afterok:$PREV_JOB script.slurm bridge/domain/model/test_single_local_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE $ws "$OUTPUT_CSV")
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
done

echo ""
echo "All sliding window jobs submitted!"
echo "Final job ID: $PREV_JOB"
echo "Monitor progress with: squeue -u \$USER"
echo "Check results with: cat $OUTPUT_CSV"
