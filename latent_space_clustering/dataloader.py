import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocess_data(self, data):
        # Example preprocessing steps
        data = data.dropna()
        return data

# Example usage
# dataloader = DataLoader('path/to/data.csv')
# data = dataloader.load_data()
# preprocessed_data = dataloader.preprocess_data(data)
