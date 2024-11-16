import pandas as pd
import h5py
import numpy as np

class DSPrepper:

    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dataset = None

    def load_and_process(self):
        self.dataset = pd.read_csv(self.dataset_path)
        self._transform_encodings()
        self.dataset.to_csv(self.output_path)
        print("Saved csv file")

    def _transform_encodings(self):
        def parse_encoding(x):
            try:
                return [float(value) for value in x.split(":")]
            except ValueError:
                return None
        self.dataset["global_encoding"] = self.dataset["global_encoding"].apply(parse_encoding)
        self.dataset.dropna(subset=["global_encoding"], inplace=True)
        self.dataset = self.dataset[["word_raw", "global_encoding"]]

    def convert_to_h5(self, h5_path: str):
        embeddings = np.array(self.dataset["global_encoding"].tolist(), dtype=np.float32)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings)
