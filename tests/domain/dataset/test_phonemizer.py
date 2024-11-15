import pytest
from src.domain.dataset import CUDADict
from traindata import Traindata
from src.domain.dataset import Phonemizer
import torch


@pytest.fixture
def phonemizer_instance():
    wordlist = ["hello", "world"]
    return Phonemizer(wordlist)


def test_initialization(phonemizer_instance):
    # Check if PAD token is set correctly
    assert phonemizer_instance.PAD == 33
    # Check if traindata, enc_inputs, dec_inputs, and targets are dictionaries
    assert isinstance(phonemizer_instance.enc_inputs, dict)
    assert isinstance(phonemizer_instance.dec_inputs, dict)
    assert isinstance(phonemizer_instance.targets, dict)


def test_len(phonemizer_instance):
    # Verify the __len__ method returns 34 as expected
    assert phonemizer_instance.get_vocabulary_size() == 34


def test_encode_valid_wordlist(phonemizer_instance):
    encoded_data = phonemizer_instance.encode(["hello"])
    # Check if the result is of type CUDADict and contains expected keys
    assert isinstance(encoded_data, CUDADict)
    assert "enc_input_ids" in encoded_data
    assert "dec_input_ids" in encoded_data
    assert "targets" in encoded_data
    assert isinstance(encoded_data["enc_input_ids"], list)
    assert isinstance(encoded_data["dec_input_ids"], list)
    assert encoded_data["enc_pad_mask"].shape[0] == len(encoded_data["enc_input_ids"])
    assert encoded_data["dec_pad_mask"].shape[0] == len(encoded_data["dec_input_ids"])
    assert encoded_data["targets"].shape[0] == len(encoded_data["dec_input_ids"])


def test_encode_word_not_found(phonemizer_instance):
    # Check if encoding a word not in the dictionary returns None
    assert phonemizer_instance.encode(["unknown_word"]) is None


def test_encode_invalid_input(phonemizer_instance):
    # Check if encode raises a TypeError when input is not a list
    with pytest.raises(TypeError):
        phonemizer_instance.encode(1)


def test_decode(phonemizer_instance):
    tokens = [1, 2, 3]
    decoded_output = phonemizer_instance.decode(tokens)
    # Verify the decoded output has correct shape and one-hot encoding
    assert decoded_output.shape == (len(tokens), 33)
    for i, token in enumerate(tokens):
        assert decoded_output[i, token] == 1


def test_padding_and_masks(phonemizer_instance):
    encoded_data = phonemizer_instance.encode(["hello", "world"])
    max_length = max(len(phonemizer_instance.enc_inputs[word]) for word in ["hello", "world"])

    # Check that enc_input_ids and dec_input_ids are padded to max_length
    for epv in encoded_data["enc_input_ids"]:
        # Ensure length matches max_length
        assert len(epv) == max_length
        # Verify that the padding section at the end has only PAD tokens
        padding_section = epv[len(epv) - (max_length - len(epv)) :]
        for val in padding_section:
            assert (
                val.item() == phonemizer_instance.PAD
            ), f"Expected padding token {phonemizer_instance.PAD}, but got {val.item()}"

    for dpv in encoded_data["dec_input_ids"]:
        assert len(dpv) == max_length - 1
        padding_section = dpv[len(dpv) - (max_length - 1 - len(dpv)) :]
        for val in padding_section:
            assert (
                val.item() == phonemizer_instance.PAD
            ), f"Expected padding token {phonemizer_instance.PAD}, but got {val.item()}"

    # Verify the shape of the padding masks
    assert encoded_data["enc_pad_mask"].shape == (len(encoded_data["enc_input_ids"]), max_length)
    assert encoded_data["dec_pad_mask"].shape == (len(encoded_data["dec_input_ids"]), max_length - 1)
