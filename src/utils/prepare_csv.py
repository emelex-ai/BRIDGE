import os
import glob
import pandas as pd

# Script to concatenate all pretraining csvs into a combined dataset for training.
extension = 'csv'
all_filenames = [i for i in glob.glob('data/pretraining/csvs/*.{}'.format(extension))]
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')