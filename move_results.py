import os

import subprocess

for i in range(1, 23):
    source_bucket = f"gs://bridge-ml-training/finetuning/{i}/*"
    dest_bucket = f"gs://bridge-ml-training/finetuning/07-02-25-10x/{i}/"
    subprocess.run(["gsutil", "mv", source_bucket, dest_bucket], shell=True)



local_data_folder = "dockerdata"
gcs_base_path = "gs://bridge-ml-training/finetuning"

if not os.path.isdir(local_data_folder):
    print(f"Error: Directory '{local_data_folder}' not found.")
else:
    for filename in os.listdir(local_data_folder):
        if filename.endswith(".csv"):
            source_path = os.path.join(local_data_folder, filename)
            file_number = os.path.splitext(filename)[0]
            dest_folder = f"{gcs_base_path}/{file_number}/"

            print(f"Uploading {source_path} to {dest_folder}")

            # Use 'gsutil cp' to upload the file.
            # check=True will raise an exception if the command fails.
            subprocess.run(["gsutil", "cp", source_path, dest_folder], shell=True, check=True)
