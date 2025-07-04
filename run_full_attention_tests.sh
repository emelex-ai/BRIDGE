#!/bin/bash

# Configuration
SEQ_LENS=(512 1024 2048 4096 8192)
D_MODEL=512
NHEAD=8
BATCH_SIZE=4
MODES=("test" "train")

echo "Starting full attention tests for both TEST and TRAIN modes..."

# Run tests for both modes
for mode in "${MODES[@]}"; do
    OUTPUT_CSV="full_attention_${mode}_results.csv"
    
    # Remove existing CSV file
    rm -f $OUTPUT_CSV
    
    echo ""
    echo "=== Running ${mode^^} mode tests ==="
    echo "Results will be saved to: $OUTPUT_CSV"
    
    # Submit jobs sequentially with dependencies for this mode
    PREV_JOB=""
    
    for seq_len in "${SEQ_LENS[@]}"; do
        echo "Submitting ${mode} job for seq_len=$seq_len"
        
        if [ -z "$PREV_JOB" ]; then
            # First job - no dependency
            JOB_ID=$(./submit_script.sh script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE "$OUTPUT_CSV" "$mode")
        else
            # Subsequent jobs - depend on previous job
            JOB_ID=$(./submit_script.sh --dependency=afterok:$PREV_JOB script.slurm bridge/domain/model/test_single_full_attention.py $seq_len $D_MODEL $NHEAD $BATCH_SIZE "$OUTPUT_CSV" "$mode")
        fi
        
        # Check if a valid job ID was returned
        if [[ -n "$JOB_ID" && "$JOB_ID" -gt 0 ]]; then
            echo "  Submitted ${mode} job $JOB_ID"
            PREV_JOB=$JOB_ID
        else
            echo "  ERROR: Failed to submit ${mode} job for seq_len=$seq_len. Aborting."
            # Optional: Cancel previously submitted jobs in the chain
            if [ -n "$PREV_JOB" ]; then
                scancel "$PREV_JOB"
                echo "  Canceled ${mode} job chain."
            fi
            exit 1
        fi
    done
    
    echo "All ${mode} jobs submitted! Final job ID: $PREV_JOB"
done

echo ""
echo "=========================================="
echo "All full attention jobs submitted for both modes!"
echo "Monitor progress with: squeue -u \$USER"
echo "Check test results with: cat full_attention_test_results.csv"
echo "Check train results with: cat full_attention_train_results.csv"
echo "=========================================="
