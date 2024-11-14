import pytest
from src.domain.dataset import CharacterTokenizer


def test_character_tokenizer_initialization():
    # Setup
    characters = ["a", "b", "c"]

    # Action
    tokenizer = CharacterTokenizer(characters)

    # Assert
    expected_vocab = ["[BOS]", "[EOS]", "[CLS]", "[UNK]", "[PAD]", "a", "b", "c"]
    assert tokenizer.vocab == expected_vocab
    assert tokenizer.char_2_idx == {ch: idx for idx, ch in enumerate(expected_vocab)}
    assert tokenizer.idx_2_char == {idx: ch for idx, ch in enumerate(expected_vocab)}
    assert tokenizer.get_vocabulary_size() == len(expected_vocab)


def test_character_tokenizer_encode_single_string():
    # Setup
    characters = ["h", "e", "l", "o"]
    tokenizer = CharacterTokenizer(characters)
    input_string = "hello"

    # Action
    encoded = tokenizer.encode(input_string)

    # Assert
    # Expected encoding: [BOS], 'h', 'e', 'l', 'l', 'o', [EOS], [PAD] x (max_length - len)
    # max_length = 5
    expected_enc_length = 2 + 5  # [BOS] + "hello" + [EOS] = 7
    assert encoded["enc_input_ids"].shape == (1, expected_enc_length)
    assert encoded["dec_input_ids"].shape == (1, 6)  # [BOS] + "hello" + [PAD] x (max_length - len)


def test_character_tokenizer_encode_list_of_strings():
    # Setup
    characters = ["a", "b", "c", "d"]
    tokenizer = CharacterTokenizer(characters)
    input_strings = ["ab", "cde"]

    # Action
    encoded = tokenizer.encode(input_strings)

    # Assert
    expected_enc_length = 2 + 3  # [BOS] + "ab" + [EOS] + [PAD] = 5 max_len=5
    assert encoded["enc_input_ids"][0].shape[0] == (expected_enc_length)
    assert encoded["dec_input_ids"][0].shape[0] == (4)  # [BOS] + "ab" +[PAD]

    expected_enc_length = 2 + 3  # [BOS] + "cde" + [EOS] = 4 max_len=5
    assert encoded["enc_input_ids"][1].shape[0] == (expected_enc_length)
    assert encoded["dec_input_ids"][1].shape[0] == (4)  # [BOS] + "cde"


def test_character_tokenizer_decode():
    # Setup
    characters = ["h", "e", "l", "o", "n", "e", "w"]
    tokenizer = CharacterTokenizer(characters)
    list_of_ints = [[0, 5, 10, 7, 7, 8, 1], [0, 9, 10, 11, 1, 4, 4]]

    # Action
    decoded = tokenizer.decode(list_of_ints)
    # Assert
    expected = ["hello", "new"]
    assert decoded == expected
