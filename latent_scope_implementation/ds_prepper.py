import pandas as pd
import h5py
import numpy as np
import sys
import os

module_path = os.environ["LATENT_SCOPE_PATH"]
if module_path not in sys.path:
    sys.path.append(module_path)

import latentscope as ls


class DSPrepper:

    def __init__(self, dataset_path: str, output_path: str, latent_scope_project_name : str):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.dataset = None
        self.latent_scope_project_name = latent_scope_project_name

    def load_and_process(self):
        self.dataset = pd.read_csv(self.dataset_path)
        self._transform_encodings()
        self.dataset.to_csv(self.output_path)
        print("Saved csv file")
        return self.dataset

    def _transform_encodings(self):
        def parse_encoding(x):
            try:
                return [float(value) for value in x.split(":")]
            except ValueError:
                return None
        
        self.dataset["global_encoding"] = self.dataset["global_encoding"].apply(parse_encoding)
        self.dataset.dropna(subset=["global_encoding"], inplace=True)
        self.dataset["global_encoding"] = self.dataset["global_encoding"].apply(
            lambda x: np.array(x, dtype=np.float32)
        )
        self.dataset = self.dataset[["word_raw", "global_encoding"]]

    def ingest_file(self, dataset):
        ls.init(module_path)
        print("--- --- --- INITIALIZED --- --- -- ")
        print(f"PROJECT NAME: {self.latent_scope_project_name}")
        ls.ingest(self.latent_scope_project_name, dataset, "global_encoding")
        print("--- --- --- INGESTED --- --- --- ")
        embeddings = np.array(dataset["global_encoding"].to_list())
        print("--- --- --- --- --- --- --- --- --- ---")
        ls.import_embeddings(self.latent_scope_project_name, np.array(dataset["global_encoding"].to_list()).reshape(-1, embeddings.shape[1]), text_column="global_encoding", model_id = "")
        ls.serve()


    def convert_to_h5(self, h5_path: str):
        embeddings = np.array(self.dataset["global_encoding"].tolist(), dtype=np.float32)
        with h5py.File(h5_path, "w") as hf:
            hf.create_dataset("embeddings", data=embeddings)


