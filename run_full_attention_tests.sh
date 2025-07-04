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
        JOB_ID=$(./submit_slurm.sh script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE $OUTPUT_CSV | grep -o '[0-9]\+')
    else
        # Subsequent jobs - depend on previous job
        JOB_ID=$(sbatch --dependency=afterok:$PREV_JOB script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE $OUTPUT_CSV | grep -o '[0-9]\+')
    fi
    
    echo "  Submitted job $JOB_ID"
    PREV_JOB=$JOB_ID
done

echo ""
echo "All full attention jobs submitted!"
echo "Final job ID: $PREV_JOB"
echo "Monitor progress with: squeue -u \$USER"
echo "Check results with: cat $OUTPUT_CSV"