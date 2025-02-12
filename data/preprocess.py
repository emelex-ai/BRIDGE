"""
Module containing helper functions to convert a list of words into input files for the language model.
This module processes word data, extracts phonetic and orthographic information, and prepares it for model input.
"""

import pickle
from typing import TypedDict
import logging

# Third party
import nltk
import numpy as np
import pandas as pd

# Local
from traindata import Traindata

nltk.download("cmudict")


class WordData(TypedDict):
    count: int
    phoneme: tuple[np.ndarray, np.ndarray]
    phoneme_shape: tuple[int, int]
    orthography: np.ndarray


def input_data(
    words: list[str], word_counts: dict, phonemes_path: str
) -> dict[str, WordData]:
    """Create the input file for the model

    Args:
        words (list): list of words
        word_counts (dict): a dictionary. Each key, value pair is the word and its count, respectively
    Returns:
        dict[str, WordData]: Dictionary with words as keys and their processed data as values.
        Each word's data includes count, phoneme representation, phoneme shape, and orthographic representation.
            {
                "word": {
                    count: int,
                    phoneme: tuple[np.array, np.array], # result of np.where
                    phoneme_shape: tuple[int, int]
                },
                "word2": {}
                ...
            }
    """
    data = Traindata(
        words, phonpath=phonemes_path, terminals=False, oneletter=True, verbose=False
    ).traindata

    input_data = {}
    for length in data.keys():
        for word, phon in zip(
            data[length]["wordlist"],
            data[length]["phon"],
        ):
            # store only non-zero phonemes
            # and the shape of the phoneme matrix
            phon_shp = phon.shape
            phon = np.where(phon)

            # add to dictionary
            input_data[word] = {
                "count": word_counts[word],
                "phoneme": phon,
                "phoneme_shape": phon_shp,
            }

    return input_data


def main(
    input_file="data/data.csv",
    output_file="data/input_data.pkl",
    phonemes_path="data/phonreps.csv",
):
    logging.info(f"Reading data from {input_file}")
    """
    try:
        with pd.read_csv(input_file) as df:
            words = [str(w).lower() for w in df["word_raw"]]
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        return
    """
    df = pd.read_csv(input_file)
    words = [str(w).lower() for w in df["word_raw"]]

    word_counts = {}

    # k-smoothing over frequency with k = 1
    for _, row in df.iterrows():

        count = row["count"]
        if pd.isna(count):
            count = 1
        else:
            count = count + 1

        word_counts[row["word_raw"]] = count

    logging.info("Processing words")
    data = input_data(words, word_counts, phonemes_path)

    logging.info(f"Saving processed data to {output_file}")
    try:
        with open(output_file, "wb") as f:
            pickle.dump(data, f)
    except IOError:
        logging.error(f"Error writing to {output_file}")


if __name__ == "__main__":
    # ideally we would have all the raw data sources
    # in the same folder, but this list was generated
    # by someone else
    # main()
    main("data/pretraining/csvs/combined_csv.csv", "data/pretraining/combined.pkl")
