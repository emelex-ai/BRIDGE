
import subprocess

for i in range(1, 23):
    source_bucket = f"gs://bridge-ml-training/finetuning/{i}/1/*"
    dest_bucket = f"gs://bridge-ml-training/finetuning/5-30-25/{i}/1/"
    subprocess.run(["gsutil", "mv", source_bucket, dest_bucket], shell=True)