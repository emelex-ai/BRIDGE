import pandas as pd
import pickle
from collections import Counter
from data.preprocess import input_data

# We may consider moving the WordData class from the
# preprocess.py file to a separate file and importing it
# here as well. In fact, in the future we may benefit from
# refactoring and using pydantic models for data validation.
# Particularly, because there is a massive "config" object
# that is passed around in the model right now. This could
# be a good candidate for a pydantic model.


def validate_input_data(data, word_count):
    # Perhaps we can think about the phoneme_feature_size
    # a parameter in the config.yaml file. Matt and
    # Nathan were discussing adding new phoneme features
    # to expand language and dialect support.
    phoneme_feature_size = 31
    for word, v in data.items():
        assert v["count"] == word_count[word]
        assert max(v["phoneme"][0]) + 1 == v["phoneme_shape"][0]
        assert v["phoneme_shape"][1] == phoneme_feature_size
        assert len(v["orthograph"]) == len(word)


def test_input_data():
    df = pd.read_csv("data/data.csv")
    words = df["word_raw"].tolist()
    words = [str(w).lower() for w in words]

    # reformat for input
    data = input_data(words)
    validate_input_data(data, Counter(words))


def test_reload_input_data():
    df = pd.read_csv("data/data.csv")
    words = df["word_raw"].tolist()
    words = [str(w).lower() for w in words]

    with open("data/input_data.pkl", "rb") as f:
        data = pickle.load(f)

    validate_input_data(data, Counter(words))
