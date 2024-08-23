"""
Module containing helper functions to convert a list of words into input files for the language model.
This module processes word data, extracts phonetic and orthographic information, and prepares it for model input.
"""

# Imports according to PEP8: standard library imports, related third party imports, local application/library specific imports
# Standard library
from collections import Counter
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
    orthograph: np.ndarray


def input_data(
    words: list[str], phonemes_path="data/phonreps.csv"
) -> dict[str, WordData]:
    """Create the input file for the model

    Args:
        words (list): list of words

    Returns:
        dict[str, WordData]: Dictionary with words as keys and their processed data as values.
        Each word's data includes count, phoneme representation, phoneme shape, and orthographic representation.
            {
                "word": {
                    count: int,
                    phoneme: tuple[np.array, np.array],
                    phoneme_shape: tuple[int, int],
                    orthograph: np.array
                },
                "word2": {}
                ...
            }
    """
    data = Traindata(
        words, phonpath=phonemes_path, terminals=False, oneletter=True, verbose=False
    ).traindata
    word_counts = Counter(words)

    input_data = {}
    for length in data.keys():
        for word, phon, orth in zip(
            data[length]["wordlist"],
            data[length]["phon"],
            data[length]["orth"],
        ):
            # store only non-zero phonemes
            # and the shape of the phoneme matrix
            phon_shp = phon.shape
            phon = np.where(phon)

            # remove orthographic padding
            orth = orth.flatten()
            orth = orth[orth != 0]

            # add to dictionary
            input_data[word] = {
                "count": word_counts[word],
                "phoneme": phon,
                "phoneme_shape": phon_shp,
                "orthograph": orth,
            }

    return input_data


def main(input_file="data/data.csv", output_file="data/input_data.pkl"):
    logging.info(f"Reading data from {input_file}")
    try:
        with pd.read_csv(input_file) as df:
            words = [str(w).lower() for w in df["word_raw"]]
    except FileNotFoundError:
        logging.error(f"Input file {input_file} not found")
        return

    logging.info("Processing words")
    data = input_data(words)

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
    main()
