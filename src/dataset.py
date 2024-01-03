#from Traindata.Traindata import Traindata
from traindata import Traindata

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


class CUDA_Dict(dict):
    def to(self, device):
        output = {}
        for key in self.keys():
            batches = self[key]
            if isinstance(batches, list):
                try:
                    output[key] = [
                        [val.to(device) for val in batch] for batch in batches
                    ]
                except:
                    print(f"batches = {batches}")
                    raise
            elif isinstance(batches, torch.Tensor):
                output[key] = batches.to(device)
            else:
                raise TypeError("Must be list or torch tensor")

        return output


class CharacterTokenizer:
    def __init__(self, list_of_characters):
        # Include custom tokens into vocabulary
        self.vocab = ["[BOS]", "[EOS]", "[CLS]", "[UNK]", "[PAD]"]
        self.vocab.extend(list_of_characters)

        self.char_2_idx, self.idx_2_char = {}, {}
        for i, ch in enumerate(self.vocab):
            self.char_2_idx[ch] = i
            self.idx_2_char[i] = ch
        # Reuse index from previous for loop to save computation
        # self.size = i + 1
        self.size = len(self.vocab)  # more robust

    def __len__(self):
        return self.size

    def encode(self, list_of_strings):
        assert isinstance(list_of_strings, str) or (
            isinstance(list_of_strings, list)
            and all(isinstance(string, str) for string in list_of_strings)
        )
        if isinstance(list_of_strings, str):
            list_of_strings = [list_of_strings]

        lengths = [len(string) for string in list_of_strings]
        max_length = max(lengths)

        # Padding function, puts beginning-of-string and end-of-string tokens
        # on beginning and end of string, after padding the original string
        # up to max length of string.
        # This function converts the output to a list rather than string.
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

        list_of_enc_strings = list(map(enc_pad, list_of_strings))
        list_of_dec_strings = list(map(dec_pad, list_of_strings))

        # Initiate encoder tokens tensor as tensor of zeros,
        enc_input_ids = torch.zeros(
            (len(list_of_enc_strings), 2 + max_length), dtype=torch.long
        )
        for idx, string in enumerate(list_of_enc_strings):
            for jdx, char in enumerate(string):
                enc_input_ids[idx, jdx] = self.char_2_idx.get(
                    char, 3
                )  # Default to [UNK]

        # Initiate decoder tokens tensor as tensor of zeros,
        dec_input_ids = torch.zeros(
            (len(list_of_dec_strings), 1 + max_length), dtype=torch.long
        )
        for idx, string in enumerate(list_of_dec_strings):
            for jdx, char in enumerate(string):
                dec_input_ids[idx, jdx] = self.char_2_idx.get(
                    char, 3
                )  # Default to [UNK]

        PAD_TOKEN = self.char_2_idx["[PAD]"]
        enc_pad_mask = enc_input_ids == PAD_TOKEN
        dec_pad_mask = dec_input_ids == PAD_TOKEN
        return CUDA_Dict(
            {
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": dec_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_pad_mask": dec_pad_mask.bool(),
            }
        )

    def decode(self, list_of_ints):
        outputs = [
            "".join([self.idx_2_char.get(i) for i in ints]) for ints in list_of_ints
        ]

        return outputs


# We may want to calculate all of the phonological vectors for the entire dataset ahead of time here.
# This way, we can just look up the vector instead of calculating it on the fly. That will save time
# during training.


class Phonemizer:
    def __init__(self, wordlist):
        self.PAD = 33

        # GE: Why are there more than 34 lines in phonreps.csv?
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
            # print("targets = ", targets)
            for i in range(len(targets)):
                tv = targets[i]
                targets[i] = torch.cat(
                    (
                        tv,
                        torch.tensor(
                            [[2] * 33] * (max_length - 1 - len(tv)), dtype=torch.long
                        ),
                    )
                )
                # print("len(tv) = ", len(tv))
                # tv = torch.cat((tv, torch.tensor([[2]*33]*(max_length-len(tv)))))
                # print("tv = ", tv)
                # sys.exit()
        else:
            raise TypeError("encode only accepts lists or a single string as input")

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
        # dec_pad_mask = torch.tensor([1])

        # Ensure that the number of tokens matches the number of boolean values in the mask
        assert len(enc_input_ids) == len(
            enc_pad_mask
        ), f"tokens is length {len(enc_input_ids)}, enc_pad_mask is length {len(enc_pad_mask)}. They must be equal"

        # Do we need to pad the targets? We do to convert it to a tensor which is needed for the CrossEntropy criterion
        return CUDA_Dict(
            {
                "enc_input_ids": enc_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_input_ids": dec_input_ids,
                "dec_pad_mask": dec_pad_mask.bool(),
                "targets": torch.stack(targets, 0),
            }
        )

    def decode(self, tokens):
        output = torch.zeros(len(tokens), 33)
        for i, token in enumerate(tokens):
            output[i, token] = 1

        return output


class ConnTextULDataset(Dataset):
    """
    ConnTextULDataset

    Dataset of
    For Matt's Phonological Feature Vectors, we will use (31, 32, 33) to represent ('[BOS]', '[EOS]', '[PAD]')
    """

    """
  GE: only read smaller datasets in --test mode. But then, how can I make arguments override my test parameters? 
  """

    # ----------------------------------------------------------------------
    def __init__(self, config, test=False, nb_rows=None, which_dataset=5, input_data = 'data.csv'):
        # Check cache folder. Perform this check in test suite.

        self.config = config
        self.which_dataset = which_dataset
        self.nb_rows = nb_rows
        self.read_orthographic_data(input_data)
        self.read_phonology_data()
        self.input_data = input_data

        # self.listed_words = [word for word in self.words]

        # Notice I created a tokenizer in this class.
        # We can use it to tokenize word output of __getitem__ below,
        # although I haven't implemented yet.
        list_of_characters = sorted(set([c for word in self.tmp_words for c in word]))
        self.character_tokenizer = CharacterTokenizer(list_of_characters)

        final_words = []
        self.max_orth_seq_len = 0
        self.max_phon_seq_len = 0
        for word in self.tmp_words:
            phonology = self.phonology_tokenizer.encode([word])
            if word == "" or word is None or word == []:
                continue
            if phonology:  # check if in phoneme_dict
                final_words.append(word)
                self.max_phon_seq_len = max(
                    self.max_phon_seq_len, len(phonology["enc_pad_mask"][0])
                )
                self.max_orth_seq_len = max(
                    self.max_orth_seq_len,
                    len(self.character_tokenizer.encode(word)["enc_input_ids"][0]),
                )

        self.words = final_words
        self.cmudict = self.phonology_tokenizer.traindata.cmudict
        self.convert_numeric_prediction = (
            self.phonology_tokenizer.traindata.convert_numeric_prediction
        )

    # ----------------------------------------------------------------------
    def read_orthographic_data(self, input_data):
        if not os.path.exists(CACHE_PATH):
            os.makedirs(CACHE_PATH)
            print("Create Cache folder: %s", CACHE_PATH)
        else:
            print("Cache folder: %s already exists", CACHE_PATH)

        if self.which_dataset == "all":
            file_path = os.path.join(DATA_PATH, input_data)
        else:
            file_path = os.path.join(
                CACHE_PATH, "data_test%05d.csv" % self.which_dataset
            )

        if not os.path.exists(file_path):
            # Create the file
            print(f"File {file_path} does not exist")
            dataset = pd.read_csv(
                os.path.join(DATA_PATH, input_data), nrows=self.nb_rows
            )
            if self.which_dataset != "all":
                dataset = dataset.sample(n=self.which_dataset)
            dataset.to_csv(file_path, index=False)
        else:
            # File exists
            print(f"File {file_path} exists")
            dataset = pd.read_csv(file_path)

        # Remove any NaN from the data (Gordon Erlebacher)
        dataset.dropna(inplace=True, axis="rows")

        # Do not remove duplicate words since we are performing curriculum training
        # Series of all lowercased words
        self.tmp_words = dataset["word_raw"].str.lower()

    # ----------------------------------------------------------------------
    def read_phonology_data(self):
        """
        Read pkl file if it exists, else create it
        """
        if self.which_dataset == "all":
            pkl_file_path = os.path.join(CACHE_PATH, "phonology_tokenizer.pkl")
        else:
            pkl_file_path = os.path.join(
                CACHE_PATH, "phonology_tokenizer%05d.pkl" % self.which_dataset
            )

        if os.path.exists(pkl_file_path):
            # pkl file exists
            with open(pkl_file_path, "rb") as f:
                self.phonology_tokenizer = pickle.load(f)
        else:
            # pkl file does not exist
            self.phonology_tokenizer = Phonemizer(self.tmp_words)
            with open(pkl_file_path, "wb") as f:
                pickle.dump(self.phonology_tokenizer, f)

    # ----------------------------------------------------------------------
    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # Currently phonology tokenizer expects a list of strings, so we enforce that here
        # by slicing the list of words to get a single word (in a list), or a list of words.
        # Under either circumstance, the input to self.encode must be a list of strings
        if isinstance(idx, int):
            string_input = self.words[idx : idx + 1]
        elif isinstance(idx, slice):
            string_input = self.words[idx]
        elif isinstance(idx, str):
            assert (
                idx in self.words
            ), f'"{idx}" not in list of input words (checked in dataset.words)'
            string_input = [idx]
        else:
            raise TypeError("idx must be int, slice, or string")

        # string_input must be a list of strings
        return self.encode(string_input)

    def encode(self, content_to_encode):
        if isinstance(content_to_encode, str):
            # If the content is a single word, wrap it in a list
            string_input = [content_to_encode]
        elif isinstance(content_to_encode, list):
            assert all(
                isinstance(content, str) for content in content_to_encode
            ), f"{content_to_encode} must be a string or a list of strings"
            string_input = content_to_encode

        orth_tokenized = self.character_tokenizer.encode(string_input)
        phon_tokenized = self.phonology_tokenizer.encode(string_input)

        return {"orthography": orth_tokenized, "phonology": phon_tokenized}
