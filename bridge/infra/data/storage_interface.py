import os
from google.cloud import storage

class StorageInterface:
    def __init__(self):
        # Create data directory if it doesn't exist
        if not os.path.exists("data"):
            os.makedirs("data")
        self.index = os.environ["CLOUD_RUN_TASK_INDEX"]
        bucket_path = os.environ.get("BUCKET_PATH", "bridge-ml-training")
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(bucket_path)
    
        
        
        self.target = f"models/pretraining/{self.index}.pkl"
        
    
    def get_data(self) -> str:
        # Get the blob for the current index
        data = self.bucket.blob(f"datasets/pretraining/{self.index}/data.csv")
        # Download the blob to a local file
        data.download_to_filename(f"data/pretraining_{self.index}.csv")
        return f"data/pretraining_{self.index}.csv"
    
    def get_config(self) -> str:
        # Get the blob for the current index
        config = self.bucket.blob(f"configs/pretraining/{self.index}/config.yaml")
        # Download the blob to a local file
        config.download_to_filename(f"data/config_{self.index}.yaml")
        return f"data/config_{self.index}.yaml"
    
    
    def upload_model(self, model_path: str):
        # Upload the model to the cloud storage bucket
        blob = self.bucket.blob(self.target)
        blob.upload_from_filename(model_path)
        print(f"Model uploaded to {self.target} in bucket {self.bucket.name}.")