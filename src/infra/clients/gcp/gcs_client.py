from google.cloud import storage
import pandas as pd
from io import StringIO


class GCSClient:
    """
    Google Cloud Storage utility class for uploading, downloading, and reading files.
    You can authenticate via:
      1. Setting the environment variable:
         export GOOGLE_APPLICATION_CREDENTIALS="/path/to/keyfile.json"
      2. Passing a service account JSON key file path to the constructor.
    """

    def __init__(self, project: str = None, credentials_path: str = None):
        """
        Initialize the GCS client.

        Args:
            project (str, optional): GCP project ID. If not provided, uses default from environment.
            credentials_path (str, optional): Path to a GCP service account JSON key file.
                If provided, will use this for authentication instead of environment variable.
        """
        if credentials_path:
            # Authenticate using the provided service account JSON
            self.client = storage.Client.from_service_account_json(
                credentials_path, project=project
            )
        else:
            # Authenticate via GOOGLE_APPLICATION_CREDENTIALS env var or default
            self.client = storage.Client(project=project)

    def download_file(
        self, bucket_name: str, blob_name: str, destination_file_name: str
    ) -> None:
        """
        Download a blob from a bucket to a local file.
        """
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded gs://{bucket_name}/{blob_name} to {destination_file_name}")

    def upload_file(
        self, bucket_name: str, source_file_name: str, destination_blob_name: str
    ) -> None:
        """
        Upload a local file to a GCS bucket.
        """
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(
            f"Uploaded {source_file_name} to gs://{bucket_name}/{destination_blob_name}"
        )

    def read_file(self, bucket_name: str, blob_name: str, as_text: bool = False):
        """
        Read a blob's content directly into memory without saving locally.

        Args:
            bucket_name (str): Name of the GCS bucket.
            blob_name (str): Name of the blob in the bucket.
            as_text (bool): If True, returns decoded string (utf-8). Otherwise, returns raw bytes.

        Returns:
            bytes or str: Content of the blob.
        """
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_text() if as_text else blob.download_as_bytes()

    def read_csv(
        self, bucket_name: str, blob_name: str, **read_csv_kwargs
    ) -> pd.DataFrame:
        """
        Read a CSV file in GCS directly into a pandas DataFrame without saving locally.

        Args:
            bucket_name (str): Name of the GCS bucket.
            blob_name (str): Name of the CSV blob in the bucket.
            **read_csv_kwargs: Additional keyword args passed to pandas.read_csv.

        Returns:
            pandas.DataFrame: Loaded DataFrame.
        """
        csv_text = self.read_file(bucket_name, blob_name, as_text=True)
        return pd.read_csv(StringIO(csv_text), **read_csv_kwargs)

    def exists(self, bucket_name: str, blob_name: str) -> bool:
        """
        Check if a blob exists in the specified GCS bucket.

        Args:
            bucket_name (str): Name of the GCS bucket.
            blob_name (str): Name of the blob in the bucket.

        Returns:
            bool: True if the blob exists, False otherwise.
        """
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()