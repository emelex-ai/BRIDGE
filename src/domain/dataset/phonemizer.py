from src.domain.dataset import CUDADict
from traindata import Traindata
from typing import List, Union
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)


class Phonemizer:
    def __init__(self, wordlist: List[str], phonpath: str = "data/phonreps.csv"):
        self.PAD = 33  # Token for padding

        # Initialize training data
        self.traindata = Traindata(
            wordlist,
            phonpath=phonpath,
            terminals=True,
            oneletter=True,
            verbose=False,
        ).traindata

        self.enc_inputs, self.dec_inputs, self.targets = self._prepare_data()

    def _prepare_data(self) -> Union[dict, dict, dict]:
        enc_inputs, dec_inputs, targets = {}, {}, {}

        for length, data in self.traindata.items():
            for word_num, (phon_vec_sos, phon_vec_eos) in enumerate(zip(data["phonSOS"], data["phonEOS"])):
                word = data["wordlist"][word_num]

                enc_inputs[word] = [torch.tensor(np.where(vec)[0], dtype=torch.long) for vec in phon_vec_sos] + [
                    torch.tensor([32])
                ]
                dec_inputs[word] = [torch.tensor(np.where(vec)[0], dtype=torch.long) for vec in phon_vec_sos]
                targets[word] = phon_vec_eos

        return enc_inputs, dec_inputs, targets

    def encode(self, wordlist: List[str]) -> CUDADict:

        enc_input_ids, dec_input_ids, targets = [], [], []
        max_length = 0

        for word in wordlist:
            enc_input = self.enc_inputs.get(word)
            dec_input = self.dec_inputs.get(word)
            target = self.targets.get(word)

            if enc_input is None or dec_input is None or target is None:
                logger.error(f"Word '{word}' not found in phonological dictionary.")
                return None

            enc_input_ids.append(enc_input)
            dec_input_ids.append(dec_input)
            targets.append(torch.tensor(target.copy(), dtype=torch.long))

            max_length = max(max_length, len(enc_input))


        for epv, dpv in zip(enc_input_ids, dec_input_ids):
            epv.extend([torch.tensor([self.PAD])] * (max_length - len(epv)))
            dpv.extend([torch.tensor([self.PAD])] * (max_length - 1 - len(dpv)))

        for i in range(len(targets)):
            targets[i] = torch.cat(
                (targets[i], torch.tensor([[2] * 33] * (max_length - 1 - len(targets[i])), dtype=torch.long))
            )

        enc_pad_mask = torch.tensor(
            [[all(val == torch.tensor([self.PAD])) for val in token] for token in enc_input_ids]
        )
        dec_pad_mask = torch.tensor(
            [[all(val == torch.tensor([self.PAD])) for val in token] for token in dec_input_ids]
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
