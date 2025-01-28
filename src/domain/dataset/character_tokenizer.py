from src.domain.dataset import CUDADict
from typing import Union
import logging
import torch
import string

logger = logging.getLogger(__name__)


class CharacterTokenizer:
    def __init__(self, device: str | torch.device | None = None):
        # Set device and default to CPU
        if isinstance(device, str):
            assert device in [
                "cpu",
                "cuda",
                "mps",
            ], "Device must be 'cpu', 'cuda', or 'mps'"
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )

        # Initialize vocabulary with special tokens
        self.special_tokens = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "[CLS]", "[SEP]"]
        self.vocab = self.special_tokens + list(string.printable)

        # Create mappings from characters to indices and vice versa
        self.char_2_idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx_2_char = {i: ch for i, ch in enumerate(self.vocab)}
        self.vocabulary_size = len(self.vocab)

        logger.info(
            f"CharacterTokenizer initialized with vocabulary size: {self.vocabulary_size}"
        )

    def get_vocabulary_size(self) -> int:
        return self.vocabulary_size

    def encode(self, list_of_strings: Union[str, list[str]]) -> CUDADict:

        # Ensure the input is either a string or a list of strings
        if isinstance(list_of_strings, str):
            list_of_strings = [list_of_strings]
        elif not isinstance(list_of_strings, list) or not all(
            isinstance(s, str) for s in list_of_strings
        ):
            logger.error("Input must be a string or a list of strings")
            raise TypeError("Input must be a string or a list of strings")

        max_length = max(len(s) for s in list_of_strings)

        enc_pad = (
            lambda s: ["[BOS]"]
            + list(s)
            + ["[EOS]"]
            + ["[PAD]"] * (max_length - len(s))
        )
        dec_pad = lambda s: ["[BOS]"] + list(s) + ["[PAD]"] * (max_length - len(s))

        # Create encoder-padded and decoder-padded string lists
        enc_strings = [enc_pad(s) for s in list_of_strings]
        dec_strings = [dec_pad(s) for s in list_of_strings]

        # Initialize tensors for encoded input
        enc_input_ids = torch.zeros(
            (len(enc_strings), 2 + max_length), dtype=torch.long, device=self.device
        )
        for i, enc_str in enumerate(enc_strings):
            for j, ch in enumerate(enc_str):
                enc_input_ids[i, j] = self.char_2_idx.get(ch, self.char_2_idx["[UNK]"])

        # Initialize tensors for decoder input
        dec_input_ids = torch.zeros(
            (len(dec_strings), 1 + max_length), dtype=torch.long, device=self.device
        )
        for i, dec_str in enumerate(dec_strings):
            for j, ch in enumerate(dec_str):
                dec_input_ids[i, j] = self.char_2_idx.get(ch, self.char_2_idx["[UNK]"])

        PAD_TOKEN = self.char_2_idx["[PAD]"]
        enc_pad_mask = enc_input_ids == PAD_TOKEN
        dec_pad_mask = dec_input_ids == PAD_TOKEN

        return CUDADict(
            {
                "enc_input_ids": enc_input_ids,
                "dec_input_ids": dec_input_ids,
                "enc_pad_mask": enc_pad_mask.bool(),
                "dec_pad_mask": dec_pad_mask.bool(),
            }
        )

    def decode(self, list_of_ints: list[list[int]]) -> list[str]:
        try:
            decoded_strings = [
                "".join(
                    [
                        self.idx_2_char[i]
                        for i in ints
                        if self.idx_2_char[i] not in self.special_tokens
                    ]
                )
                for ints in list_of_ints
            ]
        except KeyError as e:
            logger.error(f"Invalid index encountered during decoding: {e}")
            raise KeyError from e

        return decoded_strings
