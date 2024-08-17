import pandas as pd
import pickle
from collections import Counter
from data.preprocess import input_data


def validate_input_data(data, word_count):
    phoneme_feature_size = 31
    for word, v in data.items():
        assert v["count"] == word_count[word]
        assert v["phoneme"].shape[1] == phoneme_feature_size
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
