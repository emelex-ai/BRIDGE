from src.domain.datamodels import DatasetConfig
from src.domain.dataset import CUDADict
import torch.nn.functional as F
from typing import Union
import pandas as pd
import numpy as np
import logging
import torch


logger = logging.getLogger(__name__)


class Phonemizer:
    def __init__(self, input_data: dict, dataset_config: DatasetConfig):
        self.extra_token = {
            "BOS": dataset_config.dimension_phon_repr + 0,
            "EOS": dataset_config.dimension_phon_repr + 1,
            "PAD": dataset_config.dimension_phon_repr + 2,
        }
        self.phonemizer_dim = dataset_config.dimension_phon_repr + len(self.extra_token)
        self.max_phon_seq_len = dataset_config.max_phon_seq_len

        self.enc_inputs, self.dec_inputs, self.targets = self._prepare_data(input_data)
        self.feature_to_phone_map = None

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
                [torch.tensor([self.extra_token["BOS"]])] + word_phon + [torch.tensor([self.extra_token["EOS"]])]
            )

            # The decoder received the entire phonological vectors including the BOS token, but not the EOS token
            dec_inputs[word] = [torch.tensor([self.extra_token["BOS"]])] + word_phon

            # The target for the decoder is all phonological vectors including the EOS token, but excluding the BOS token
            targets[word] = [
                # targets are one-hot encoded (PAD token not included)
                torch.isin(torch.arange(self.phonemizer_dim - 1), phon).long()
                for phon in word_phon + [torch.tensor([self.extra_token["EOS"]])]
            ]

        return enc_inputs, dec_inputs, targets

    def encode(self, wordlist: list[str]) -> CUDADict:
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
                        [[2] * (self.phonemizer_dim - 1)] * (max_length - 1 - len(targets[i])),
                        dtype=torch.long,
                    ),
                )
            )

        enc_pad_mask = torch.tensor(
            [[all(val == torch.tensor([self.extra_token["PAD"]])) for val in token] for token in enc_input_ids]
        )
        dec_pad_mask = torch.tensor(
            [[all(val == torch.tensor([self.extra_token["PAD"]])) for val in token] for token in dec_input_ids]
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

    def decode(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        """
        Decodes a list of token tensors into a multi-hot encoded tensor.

        Each element in the input list is a tensor representing a group of tokens
        that should remain together. Any token that is equal to an extra token
        (BOS, EOS, PAD) or 33 is dropped. The remaining tokens in each group are
        one-hot encoded (with ones at each valid token index) into a 33-dimensional vector.

        For example, given:
            tokens = [
                tensor([31]),          # extra token (BOS) -> drop it
                tensor([2, 7]),        # valid tokens -> one-hot vector with ones at indices 2 and 7
                tensor([2, 14]),       # valid tokens -> one-hot vector with ones at indices 2 and 14
                tensor([6]),           # valid token  -> one-hot vector with a one at index 6
                tensor([32])           # extra token (EOS) -> drop it
            ]
        The output will be a tensor of shape (3, 33) corresponding to the three valid groups.
        """

        extra_token_values = set(self.extra_token.values())

        decoded_vectors = []
        # Process each group (tensor) in the tokens list.
        for token_group in tokens:
            # token_group is a tensor, possibly with multiple token values.
            # Convert it to a list.
            if token_group.ndim > 0:
                group_tokens = token_group.tolist()
            else:
                group_tokens = [int(token_group.item())]

            # Filter out any tokens that are extra tokens.
            group_tokens_filtered = [t for t in group_tokens if t not in extra_token_values]
            # If, after filtering, the group is empty, skip it.
            if not group_tokens_filtered:
                continue

            # Create a multi-hot vector for this group.
            one_hot = torch.zeros(31, dtype=torch.int)
            for t in group_tokens_filtered:
                one_hot[t] = 1

            decoded_vectors.append(one_hot)

        return self.features_to_phonemes(torch.stack(decoded_vectors, dim=0))

    def get_vocabulary_size(self):
        return self.phonemizer_dim

    def _load_feature_to_phone_map(self) -> dict:
        """
        Loads a mapping from a tuple of 32 binary features to the corresponding phoneme
        from the CSV file at 'data/phonreps.csv' using pandas.

        The CSV is expected to have a structure like:

            phone,labial,dental,alveolar,palatal,velar,glottal,stop,fricative,affricate,
            nasal,liquid,glide,rhotic,tap,voice,front,center,back,close,close-mid,mid,
            open-mid,near-open,open,tense,retroflex,round,post_y,post_w,primary,secondary

        The first column should be the phoneme label. The remaining columns are the features.
        """
        mapping = {}
        csv_path = "data/phonreps.csv"
        try:
            df = pd.read_csv(csv_path)
            # Remove any whitespace from column names.
            df.columns = df.columns.str.strip()
            # Assume the first column holds the phone labels.
            phone_col = df.columns[0]
            # All remaining columns are features.
            feature_cols = df.columns[1:]
            for _, row in df.iterrows():
                phone = str(row[phone_col]).strip()
                # Create a tuple of integers for the features.
                features = tuple(int(row[col]) for col in feature_cols)
                mapping[features] = phone
        except FileNotFoundError:
            logger.error(f"CSV file not found at path: {csv_path}")
        except Exception as e:
            logger.error(f"Error reading CSV file at {csv_path}: {e}")
        return mapping

    def _get_closest_phoneme(self, feature_vector: Union[torch.Tensor, list[int]]) -> str:
        """
        Calculates cosine similarity between the provided feature vector and each candidate's feature vector
        from the mapping, then returns the phone with the highest similarity.
        """
        # Convert the input feature vector to a float tensor.
        if isinstance(feature_vector, list):
            input_vec = torch.tensor(feature_vector, dtype=torch.float)
        else:
            input_vec = feature_vector.float()

        best_similarity = -1.0
        best_phone = "UNK"

        # Compare against each candidate in the mapping.
        for candidate_features, candidate_phone in self.feature_to_phone_map.items():

            candidate_vec = torch.tensor(candidate_features, dtype=torch.float)
            sim = F.cosine_similarity(input_vec, candidate_vec, dim=0)
            if sim > best_similarity:
                best_similarity = sim
                best_phone = candidate_phone

        return best_phone

    def get_phoneme_from_features(self, feature_vector: list[int]) -> str:
        """
        Given a feature vector (a one-dimensional torch.Tensor or list of 0s and 1s
        of length 32), return the corresponding phoneme (as defined by the CSV data).
        If no match is found, a warning is logged and "UNK" is returned.
        """
        if isinstance(feature_vector, torch.Tensor):
            feature_list = feature_vector.tolist()
        else:
            feature_list = feature_vector

        feature_tuple = tuple(feature_list)
        if self.feature_to_phone_map is None:
            self.feature_to_phone_map = self._load_feature_to_phone_map()

        phone = self.feature_to_phone_map.get(feature_tuple)
        if phone is None:
            logger.warning(f"No phoneme found for feature vector: {feature_tuple}. Calculating cosine similarity.")
            phone = self._get_closest_phoneme(feature_list)
        return phone

    def features_to_phonemes(self, feature_vectors: torch.Tensor) -> list[str]:
        """
        Converts a 2D tensor (shape: [seq_len, feature_dim]) of feature vectors
        into a list of phoneme strings.
        """
        return [self.get_phoneme_from_features(fv) for fv in feature_vectors]
