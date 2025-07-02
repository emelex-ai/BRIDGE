
import subprocess

for i in range(1, 23):
    source_bucket = f"gs://bridge-ml-training/finetuning/{i}/*"
    dest_bucket = f"gs://bridge-ml-training/finetuning/6-30-25/{i}/"
    subprocess.run(["gsutil", "mv", source_bucket, dest_bucket], shell=True)