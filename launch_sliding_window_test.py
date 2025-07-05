#!/usr/bin/env python3
"""
Python launcher script for sliding window attention test.

This script submits the sliding window attention test to SLURM for GPU execution.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def submit_slurm_job(script_path: str, python_script: str, *args) -> tuple[bool, str]:
    """Submit a job to SLURM.

    Args:
        script_path: Path to the SLURM script
        python_script: Python script to run
        *args: Additional arguments for the Python script

    Returns:
        Tuple of (success, job_id_or_error)
    """
    cmd = ["sbatch", script_path, python_script] + list(args)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Extract job ID from output (format: "Submitted batch job 12345")
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            return True, job_id
        else:
            return False, f"Unexpected output: {output}"

    except subprocess.CalledProcessError as e:
        return False, f"SLURM submission failed: {e.stderr}"
    except Exception as e:
        return False, f"Error: {e}"


def monitor_job(job_id: str, check_interval: int = 10) -> bool:
    """Monitor a SLURM job until completion.

    Args:
        job_id: SLURM job ID
        check_interval: How often to check job status (seconds)

    Returns:
        True if job completed successfully, False otherwise
    """
    print(f"Monitoring job {job_id}...")

    while True:
        try:
            # Check job status
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=True,
            )

            status = result.stdout.strip()

            if not status:
                # Job no longer in queue - check if it completed
                result = subprocess.run(
                    ["sacct", "-j", job_id, "-n", "-o", "State"],
                    capture_output=True,
                    text=True,
                    check=True,
                )

                final_status = result.stdout.strip().split("\n")[0]
                print(f"Job {job_id} completed with status: {final_status}")

                return "COMPLETED" in final_status

            else:
                print(f"Job {job_id} status: {status}")

                if status in ["FAILED", "CANCELLED", "TIMEOUT"]:
                    print(f"Job {job_id} failed with status: {status}")
                    return False

                # Wait before checking again
                time.sleep(check_interval)

        except subprocess.CalledProcessError as e:
            print(f"Error checking job status: {e}")
            return False
        except KeyboardInterrupt:
            print(f"\nInterrupted. Job {job_id} is still running.")
            return False


def main():
    """Main function to launch the sliding window attention test."""
    print("Sliding Window Attention Test Launcher")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("script.slurm").exists():
        print("❌ Error: script.slurm not found in current directory.")
        print("Please run this script from the BRIDGE project root.")
        sys.exit(1)

    # Check if the test script exists
    test_script = "run_sliding_window_test.py"
    if not Path(test_script).exists():
        print(f"❌ Error: {test_script} not found in current directory.")
        sys.exit(1)

    print(f"Submitting sliding window attention test to SLURM...")
    print(f"Test script: {test_script}")
    print(f"SLURM script: script.slurm")

    # Submit the job
    success, result = submit_slurm_job("script.slurm", test_script)

    if success:
        job_id = result
        print(f"✅ Job submitted successfully!")
        print(f"Job ID: {job_id}")

        # Ask if user wants to monitor the job
        try:
            monitor = input("\nMonitor job progress? (y/N): ").lower().strip()
            if monitor in ["y", "yes"]:
                success = monitor_job(job_id)

                if success:
                    print("✅ Job completed successfully!")

                    # Try to show the output
                    output_file = f"slurm-apptainer_python_job-{job_id}.out"
                    if Path(output_file).exists():
                        print(f"\n📄 Job output ({output_file}):")
                        print("-" * 50)
                        with open(output_file, "r") as f:
                            print(f.read())

                else:
                    print("❌ Job failed or was cancelled.")

                    # Try to show the error
                    error_file = f"slurm-apptainer_python_job-{job_id}.err"
                    if Path(error_file).exists():
                        print(f"\n📄 Job error ({error_file}):")
                        print("-" * 50)
                        with open(error_file, "r") as f:
                            print(f.read())
            else:
                print(f"Job {job_id} is running. Check status with: squeue -j {job_id}")
                print(
                    f"View output with: tail -f slurm-apptainer_python_job-{job_id}.out"
                )

        except KeyboardInterrupt:
            print(f"\nJob {job_id} is still running.")

    else:
        print(f"❌ Failed to submit job: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
