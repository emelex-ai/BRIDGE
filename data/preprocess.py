"""
Module containing help functionssss to convert a list of word into the input files for the model
"""

from collections import Counter
import nltk
import numpy as np
import pandas as pd
import pickle
from traindata import Traindata

nltk.download("cmudict")


def input_data(words: list) -> dict:
    """Create the input file for the model

    Args:
        words (list): list of words

    Returns:
        dict: dictionary with the following structure:
            {
                "word": {
                    count: int,
                    phoneme: np.array,
                    orthograph: np.array
                },
                "word2": {}
                ...
            }
    """
    phonemes_path = "data/phonreps.csv"
    data = Traindata(
        words, phonpath=phonemes_path, terminals=False, oneletter=True, verbose=False
    ).traindata
    word_count = Counter(words)

    input_data = {}
    for length in data.keys():
        for word, phon, orth in zip(
            data[length]["wordlist"],
            data[length]["phon"],
            data[length]["orth"],
        ):
            # don't keep orthograph padding
            orth = orth.flatten()
            orth = orth[orth != 0]

            phon_shp = phon.shape
            phon = np.where(phon)

            # add to dictionary
            input_data[word] = {
                "count": word_count[word],
                "phoneme": phon,
                "phoneme_shape": phon_shp,
                "orthograph": orth,
            }
    return input_data


if __name__ == "__main__":
    # ideally we would have all the raw data sources
    # in the same folder, but this list was generated
    # by someone else
    df = pd.read_csv("data/data.csv")
    words = df["word_raw"].tolist()
    words = [str(w).lower() for w in words]

    # reformat for input
    data = input_data(words)

    # pickle
    with open("data/input_data.pkl", "wb") as f:
        pickle.dump(data, f)
