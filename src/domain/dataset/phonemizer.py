from src.domain.dataset import CUDADict
from typing import List, Union
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)


class Phonemizer:
    def __init__(self, input_data: dict, dimension_phon_repr: int):
        self.phoneme_reps_dim = dimension_phon_repr
        self.extra_token = {
            "BOS": self.phoneme_reps_dim + 0,
            "EOS": self.phoneme_reps_dim + 1,
            "PAD": self.phoneme_reps_dim + 2,
        }
        self.phonemizer_dim = self.phoneme_reps_dim + len(self.extra_token)

        self.enc_inputs, self.dec_inputs, self.targets = self._prepare_data(input_data)

    def _prepare_data(self, input_data) -> Union[dict, dict, dict]:
        enc_inputs, dec_inputs, targets = {}, {}, {}

        for word, data in input_data.items():
            # Reconstruct phonemes from data in the input file (compressed with np.where)
            # For example, for the word test, the phoneme tuple in input_data["test"] is
            # 'phoneme': (array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3]),
            #             array([ 2,  6, 14, 15, 21, 23, 24, 29,  2,  7,  2,  6]))
            # which represents a total of 4 phonemes, resulting in the following word representation
            # word_phon = [tensor([2, 6]), tensor([14, 15, 21, 23, 24, 29]), tensor([2, 7]), tensor([2, 6])]
            word_phon = []
            phoneme_index = data["phoneme"][0]
            phoneme_features = data["phoneme"][1]
            for i in np.unique(phoneme_index):
                mask = phoneme_index == i
                word_phon.append(torch.tensor(phoneme_features[mask], dtype=torch.long))

            # The encoder receives the entire phonological vector with the BOS and EOS tokens
            enc_inputs[word] = (
                [torch.tensor([self.extra_token["BOS"]])]
                + word_phon
                + [torch.tensor([self.extra_token["EOS"]])]
            )
            print(word)
            print(enc_inputs[word])

            # The decoder received the entire phonological vectors including the BOS token, but not the EOS token
            dec_inputs[word] = [
                torch.tensor([self.extra_token["BOS"]])
            ] + word_phon
            print(dec_inputs[word])

            # The target for the decoder is all phonological vectors including the EOS token, but excluding the BOS token
            targets[word] = [
                # targets are one-hot encoded (PAD token not included)
                torch.isin(torch.arange(self.phonemizer_dim - 1), phon).long()
                for phon in word_phon + [torch.tensor([self.extra_token["EOS"]])]
            ]
            print(targets[word])
            exit()

        return enc_inputs, dec_inputs, targets

    def encode(self, wordlist: List[str]) -> CUDADict:
        enc_input_ids, dec_input_ids, targets = [], [], []
        max_length = 0

        for word in wordlist:
            enc_input = self.enc_inputs.get(word)
            dec_input = self.dec_inputs.get(word)
            target = self.targets.get(word)

            if enc_input is None or dec_input is None or target is None:
                # logger.error(f"Word '{word}' not found in phonological dictionary.")
                return None

            enc_input_ids.append(enc_input)
            dec_input_ids.append(dec_input)
            targets.append(target)

            max_length = max(max_length, len(enc_input))

        for epv, dpv in zip(enc_input_ids, dec_input_ids):
            epv.extend([torch.tensor([self.extra_token["PAD"]])] * (max_length - len(epv)))
            dpv.extend([torch.tensor([self.extra_token["PAD"]])] * (max_length - 1 - len(dpv)))

        for i in range(len(targets)):
            targets[i] = torch.cat(
                (
                    torch.stack(targets[i]),
                    torch.tensor(
                        [[2] * (self.phonemizer_dim - 1)]
                        * (max_length - 1 - len(targets[i])),
                        dtype=torch.long,
                    ),
                )
            )

        enc_pad_mask = torch.tensor(
            [
                [all(val == torch.tensor([self.extra_token["PAD"]])) for val in token]
                for token in enc_input_ids
            ]
        )
        dec_pad_mask = torch.tensor(
            [
                [all(val == torch.tensor([self.extra_token["PAD"]])) for val in token]
                for token in dec_input_ids
            ]
        )

        return CUDADict(
            {
                "enc_input_ids": enc_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_input_ids": dec_input_ids,
                "dec_pad_mask": dec_pad_mask.bool(),
                "targets": torch.stack(targets, 0),
            }
        )

    def decode(self, tokens: List[int]) -> torch.Tensor:
        logger.info(f"Decoding tokens: {tokens}")
        output = torch.zeros(len(tokens), 33)
        for i, token in enumerate(tokens):
            output[i, token] = 1
        return output

    def get_vocabulary_size(self):
        return self.phonemizer_dim
