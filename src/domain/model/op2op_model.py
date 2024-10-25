from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model, Encoder, Decoder
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class OP2OPModel(Model):

    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(OP2OPModel, self).__init__(model_config, dataset_config)


    def forward(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Encode orthography and phonology
        orthography_encoding = self.encode_orthography(orth_enc_input, orth_enc_pad_mask)
        phonology_encoding = self.encode_phonology(phon_enc_input, phon_enc_pad_mask)

        # Add global embedding
        global_embedding = self.global_embedding.repeat(orthography_encoding.shape[0], 1, 1)
        orthography_encoding = torch.cat((global_embedding, orthography_encoding), dim=1)
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)

        # Mix encodings
        mixed_encoding = orthography_encoding[:, : self.d_embedding] + phonology_encoding[:, : self.d_embedding]

        # Decode orthography
        orth_dec_input = self.embed_orth_tokens(orth_dec_input)
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1], orth_dec_input.device)
        orth_output = self.decode(
            self.orthography_decoder, orth_dec_input, mixed_encoding, orth_ar_mask, orth_dec_pad_mask
        )
        orth_token_logits = self.linear_orthography_decoder(orth_output)

        # Decode phonology
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1], phon_dec_input.device)
        phon_output = self.decode(
            self.phonology_decoder, phon_dec_input, mixed_encoding, phon_ar_mask, phon_dec_pad_mask
        )
        phon_token_logits = self.linear_phonology_decoder(phon_output)

        return {"orth": orth_token_logits, "phon": phon_token_logits}
