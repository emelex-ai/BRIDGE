from Traindata.Traindata import Traindata

# import Traindata.Traindata
from torch.utils.data import Dataset
import pandas as pd
import torch
import nltk
import numpy as np
import pickle
import os

nltk.download("cmudict")


DATA_PATH = "./data/"
# The user can remove the contents of the cache at any time.
# The user can remove the cache folder, which will auto-generate
CACHE_PATH = os.path.join(DATA_PATH, ".cache")

"""
Class Name: CUDA_Dict
Class Parameters: device
    'device' can represent eother a GPU or other hardware where to move data.
Class Purpose: CUDA_Dict repesents a way to take existing input and move it to the device that is specified.
"""
class CUDA_Dict(dict):
    def to(self, device):
        output = {}
        for key in self.keys():
            batches = self[key]
            if isinstance(batches, list):
                try:
                    # If the batches are a list, move each tensor in the batch to the specified device
                    output[key] = [
                        [val.to(device) for val in batch] for batch in batches
                    ]
                except:
                    # If an error occurs, print the batches and raise the error
                    print(f"batches = {batches}")
                    raise
            elif isinstance(batches, torch.Tensor):
                # If the batches are a single tensor, move it to the specified device
                output[key] = batches.to(device)
            else:
                # If the batches are neither a list nor a tensor, raise a TypeError
                raise TypeError("Must be list or torch tensor")

        # Return the output dictionary
        return output



"""
Class Name: CharacterTokenizer
Class Parameters: list_of_characters 
    'list_of_characters' represents custom tokens that can be included into the default vocabulary. 
Class Purpose: The function of this class is both to encode or decode characters presented to it. Using its function 'encode', it encodes the character returning a CUDA dictionary.
With the function 'decode', it returns a list of integers to then be decoded.
"""
class CharacterTokenizer:
    def __init__(self, list_of_characters):
        # Initialize the vocabulary with predefined tokens
        self.vocab = ["[BOS]", "[EOS]", "[CLS]", "[UNK]", "[PAD]"]
        # Add custom characters to the vocabulary
        self.vocab.extend(list_of_characters)

        # Create two dictionaries for mapping characters to indices and vice versa
        self.char_2_idx, self.idx_2_char = {}, {}
        for i, ch in enumerate(self.vocab):
            self.char_2_idx[ch] = i
            self.idx_2_char[i] = ch

        # Set the size of the vocabulary
        self.size = len(self.vocab)

    def __len__(self):
        # Return the size of the vocabulary
        return self.size

    def encode(self, list_of_strings):
        # Check if the input is a string or a list of strings
        assert isinstance(list_of_strings, str) or (
            isinstance(list_of_strings, list)
            and all(isinstance(string, str) for string in list_of_strings)
        )
        # If the input is a string, convert it to a list of strings
        if isinstance(list_of_strings, str):
            list_of_strings = [list_of_strings]

        # Calculate the length of each string in the list
        lengths = [len(string) for string in list_of_strings]
        # Find the maximum length among all strings
        max_length = max(lengths)

        # Define functions for padding the strings
        # The padding function adds beginning-of-string and end-of-string tokens
        # to the beginning and end of the string, respectively.
        # The string is then padded up to the maximum length of the string.
        # The output is a list rather than a string.
        enc_pad = (
            lambda string: ["[BOS]"]
            + list(string)
            + ["[EOS]"]
            + (max_length - len(string)) * ["[PAD]"]
        )
        dec_pad = (
            lambda string: ["[BOS]"]
            + list(string)
            + (max_length - len(string)) * ["[PAD]"]
        )

        # Apply the padding function to each string in the list
        list_of_enc_strings = list(map(enc_pad, list_of_strings))
        list_of_dec_strings = list(map(dec_pad, list_of_strings))

        # Initialize the encoder tokens tensor as a tensor of zeros
        enc_input_ids = torch.zeros(
            (len(list_of_enc_strings), 2 + max_length), dtype=torch.long
        )
        # Convert each character in each string to its corresponding index
        for idx, string in enumerate(list_of_enc_strings):
            for jdx, char in enumerate(string):
                enc_input_ids[idx, jdx] = self.char_2_idx.get(
                    char, 3
                )  # Default to [UNK]

        # Initialize the decoder tokens tensor as a tensor of zeros
        dec_input_ids = torch.zeros(
            (len(list_of_dec_strings), 1 + max_length), dtype=torch.long
        )
        # Convert each character in each string to its corresponding index
        for idx, string in enumerate(list_of_dec_strings):
            for jdx, char in enumerate(string):
                dec_input_ids[idx, jdx] = self.char_2_idx.get(
                    char, 3
                )  # Default to [UNK]

        # Get the index of the [PAD] token
        PAD_TOKEN = self.char_2_idx["[PAD]"]
        # Create masks for the encoder and decoder tokens
        enc_pad_mask = enc_input_ids == PAD_TOKEN
        dec_pad_mask = dec_input_ids == PAD_TOKEN

        # Return a dictionary containing the encoder and decoder tensors and their corresponding masks
        return CUDA_Dict(
            {
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": dec_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_pad_mask": dec_pad_mask.bool(),
            }
        )

    def decode(self, list_of_ints):
        # Convert each integer in the list to its corresponding character
        outputs = [
            "".join([self.idx_2_char.get(i) for i in ints]) for ints in list_of_ints
        ]

        # Return the list of outputs
        return outputs


# We may want to calculate all of the phonological vectors for the entire dataset ahead of time here.
# This way, we can just look up the vector instead of calculating it on the fly. That will save time
# during training.

"""
Class Name: Phonemizer
Class Parameters: wordlist
    'wordlist' represents the list of words to be turned into phonems through the phonemizer class and the model as a whole.
Class Purpose: Phonemizer is a class that is used to calculate all of the phonological vectors that may be present in the dataset ahead of time. 
If the vectors are already present, then they won't have to be calculated during training, saving much time.
"""
class Phonemizer:
    def __init__(self, wordlist):
        self.PAD = 33

        # Load the training data from a CSV file
        self.traindata = Traindata(
            wordlist,
            phonpath="raw/phonreps.csv",
            terminals=True,
            oneletter=True,
            verbose=False,
        )
        traindata = self.traindata.traindata
        self.enc_inputs = {}
        self.dec_inputs = {}
        self.targets = {}
        for length in traindata.keys():
            for word_num, (phon_vec_sos, phon_vec_eos) in enumerate(
                zip(traindata[length]["phonSOS"], traindata[length]["phonEOS"])
            ):
                word = traindata[length]["wordlist"][word_num]
                # The encoder receives the entire phonological vector include the BOS and EOS tokens
                self.enc_inputs[word] = [
                    torch.tensor(np.where(vec)[0], dtype=torch.long)
                    for vec in phon_vec_sos
                ] + [
                    torch.tensor([32])
                ]  # 32 is the EOS token location
                # The decoder received the entire phonological vectors including the BOS token, but not the EOS token
                self.dec_inputs[word] = [
                    torch.tensor(np.where(vec)[0], dtype=torch.long)
                    for vec in phon_vec_sos
                ]
                # The target for the decoder is all phonological vectors including the EOS token, but excluding the BOS token
                self.targets[word] = phon_vec_eos

        del traindata

    def __len__(self):
        # The size of the vocabulary is fixed at 34
        return 34

    def encode(self, wordlist):
        enc_input_ids = []
        dec_input_ids = []
        targets = []

        if isinstance(wordlist, list):
            max_length = 0
            for word in wordlist:
                # Make sure all words are in the phonological dictionary
                enc_input = self.enc_inputs.get(word, None)
                dec_input = self.dec_inputs.get(word, None)
                target = self.targets.get(word, None)
                # If any word is not in the dictionary, skip the batch
                if enc_input is None or dec_input is None or target is None:
                    return None
                # Collect all token lists in a larger list while calculating the max length of this batch
                enc_input_ids.append(enc_input.copy())
                dec_input_ids.append(dec_input.copy())
                targets.append(torch.tensor(target.copy(), dtype=torch.long))
                # All three, enc_input, dec_input, and target should be the same length. So all share the same max_length
                # (though we subtract 1 from the decoder input and targets because the BOS/EOS tokens were removed)
                max_length = max(max_length, len(enc_input))
            # Now that we know the max length of this batch, we pad the encoder and decoder input token list with PAD tokens
            for epv, dpv in zip(enc_input_ids, dec_input_ids):
                epv.extend([torch.tensor([self.PAD])] * (max_length - len(epv)))
                dpv.extend([torch.tensor([self.PAD])] * (max_length - 1 - len(dpv)))
            # We then include padding, or indices in the targets to be passed to the 'ignore_index' parameter in the CrossEntropyLoss
            # Since each phonological vector is either on or off, it is a binary classification problem, so valid labels are either 0, or 1.
            # We will include labels of '2' where the padding is in the target vectors
            for i in range(len(targets)):
                # Append the padding tokens to the targets
                tv = targets[i]
                targets[i] = torch.cat(
                    (
                        targets[i],
                        torch.tensor(
                            [[2] * 33] * (max_length - 1 - len(targets[i])), dtype=torch.long
                        ),
                    )
                )
            # Create masks for the encoder and decoder tokens
            enc_pad_mask = torch.tensor(
                [
                    [all(val == torch.tensor([self.PAD])) for val in token]
                    for token in enc_input_ids
                ]
            )
            dec_pad_mask = torch.tensor(
                [
                    [all(val == torch.tensor([self.PAD])) for val in token]
                    for token in dec_input_ids
                ]
            )
            # Ensure that the number of tokens matches the number of boolean values in the mask
            assert len(enc_input_ids) == len(
                enc_pad_mask
            ), f"tokens is length {len(enc_input_ids)}, enc_pad_mask is length {len(enc_pad_mask)}. They must be equal"

            # Return a dictionary containing the encoder and decoder tensors, their corresponding masks, and the targets
            return CUDA_Dict(
                {
                    "enc_input_ids": enc_input_ids,
                    "enc_pad_mask": enc_pad_mask.bool(),
                    "dec_input_ids": dec_input_ids,
                    "dec_pad_mask": dec_pad_mask.bool(),
                    "targets": torch.stack(targets, 0),
                }
            )
        else:
            raise TypeError("encode only accepts lists or a single string as input")

    def decode(self, tokens):
        # Initialize an output tensor of zeros
        output = torch.zeros(len(tokens), 33)
        # For each token in the input, set the corresponding index in the output tensor to 1
        for i, token in enumerate(tokens):
            output[i, token] = 1

        # Return the output tensor
        return output

"""
Class Name: ConnTextULDataset
Class Purpose: This class is designed to handle and process a dataset for the Emelex Deep Learning model. It reads and prepares orthographic and phonological data, tokenizes the data, and provides methods to access the data.
Class Parameters:
    config (dict): A dictionary containing configuration parameters for the dataset.
    test (bool): A boolean indicating whether the dataset is for testing. Default is False.
    nb_rows (int): The number of rows to read from the dataset. Default is None, which means all rows will be read.
    which_dataset (int): The index of the dataset to use. Default is 5.

"""
class ConnTextULDataset(Dataset):
    def __init__(self, config, test=False, nb_rows=None, which_dataset=5):
        # Store the configuration and dataset parameters
        self.config = config
        self.which_dataset = which_dataset
        self.nb_rows = nb_rows

        # Read the orthographic and phonological data
        self.read_orthographic_data()
        self.read_phonology_data()

        # Create a character tokenizer
        list_of_characters = sorted(set([c for word in self.tmp_words for c in word]))
        self.character_tokenizer = CharacterTokenizer(list_of_characters)

        # Initialize variables for the maximum length of the orthographic and phonological sequences
        self.max_orth_seq_len = 0
        self.max_phon_seq_len = 0

        # Initialize a list for the final words
        final_words = []

        # For each word in the temporary words
        for word in self.tmp_words:
            # Encode the phonology of the word
            phonology = self.phonology_tokenizer.encode([word])

            # If the word is empty, None, or an empty list, skip this iteration
            if word == "" or word is None or word == []:
                continue

            # If the word is in the phoneme dictionary
            if phonology:
                # Add the word to the final words
                final_words.append(word)

                # Update the maximum length of the phonological sequences
                self.max_phon_seq_len = max(
                    self.max_phon_seq_len, len(phonology["enc_pad_mask"][0])
                )

                # Update the maximum length of the orthographic sequences
                self.max_orth_seq_len = max(
                    self.max_orth_seq_len,
                    len(self.character_tokenizer.encode(word)["enc_input_ids"][0]),
                )

        # Store the final words
        self.words = final_words

    def read_orthographic_data(self):
        # Check if the cache folder exists and create it if it doesn't
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
            print("Create Cache folder: %s", CACHE_PATH)
        else:
            print("Cache folder: %s already exists", CACHE_PATH)

        # Determine the file path based on the dataset
        if self.which_dataset == "all":
            file_path = os.path.join(DATA_PATH, "data.csv")
        else:
            file_path = os.path.join(CACHE_PATH, "data_test%05d.csv" % self.which_dataset)

        # If the file doesn't exist, create it
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist")
            dataset = pd.read_csv(os.path.join(DATA_PATH, "data.csv"), nrows=self.nb_rows)
            if self.which_dataset != "all":
                dataset = dataset.sample(n=self.which_dataset)
            dataset.to_csv(file_path, index=False)
        else:
            print(f"File {file_path} exists")
            dataset = pd.read_csv(file_path)

        # Store the words in lowercase
        self.tmp_words = dataset["word_raw"].str.lower()

    # ----------------------------------------------------------------------
    def read_phonology_data(self):
        # Determine the file path for the pickle file based on the dataset
        if self.which_dataset == "all":
            pkl_file_path = os.path.join(CACHE_PATH, "phonology_tokenizer.pkl")
        else:
            pkl_file_path = os.path.join(
                CACHE_PATH, "phonology_tokenizer%05d.pkl" % self.which_dataset
            )

        # If the pickle file exists, load the phonology tokenizer
        if os.path.exists(pkl_file_path):
            with open(pkl_file_path, "rb") as f:
                self.phonology_tokenizer = pickle.load(f)
        else:
            # If the pickle file doesn't exist, create a new phonology tokenizer
            self.phonology_tokenizer = Phonemizer(self.tmp_words)
            # Save the phonology tokenizer to a pickle file
            with open(pkl_file_path, "wb") as f:
                pickle.dump(self.phonology_tokenizer, f)

    def __len__(self):
        # Return the number of words in the dataset
        return len(self.words)

    def __getitem__(self, idx):
        # Get the word at the specified index
        string_input = self.words[idx]

        # Tokenize the orthographic and phonological representations of the word
        orth_tokenized = self.character_tokenizer.encode(string_input)
        phon_tokenized = self.phonology_tokenizer.encode(string_input)

        # Return a dictionary containing the orthographic and phonological tokenized representations
        return {"orthography": orth_tokenized, "phonology": phon_tokenized}
