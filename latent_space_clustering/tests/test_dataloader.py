import unittest
import pandas as pd
from latent_space_clustering.dataloader import DataLoader

class TestDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a small test dataframe
        cls.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        # Save the test dataframe to a csv file
        cls.test_data.to_csv('tests/test_data.csv', index=False)

    def test_load_data(self):
        dataloader = DataLoader('tests/test_data.csv')
        data = dataloader.load_data()
        self.assertIsNotNone(data)
        self.assertEqual(data.shape, self.test_data.shape)

    def test_preprocess_data(self):
        dataloader = DataLoader('tests/test_data.csv')
        data = dataloader.load_data()
        preprocessed_data = dataloader.preprocess_data(data)
        self.assertFalse(preprocessed_data.isnull().values.any())

if __name__ == '__main__':
    unittest.main()