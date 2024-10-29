from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model, Encoder, Decoder
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class P2OModel(Model):

    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(P2OModel, self).__init__(model_config, dataset_config)

    def embed_p2o(self, phonology, phonology_padding_mask):
        phonology = self.embed_phon_tokens(phonology)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phonology_padding_mask)
        global_embedding = self.global_embedding.repeat(phonology_encoding.shape[0], 1, 1)
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        phonology_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (phonology_encoding.shape[0], self.d_embedding),
                    device=phonology_encoding.device,
                    dtype=torch.bool,
                ),
                phonology_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(
            phonology_encoding, src_key_padding_mask=phonology_encoding_padding_mask
        )

        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

    def forward(
        self,
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        mixed_encoding = self.embed_p2o(phon_enc_input, phon_enc_pad_mask)
        orth_dec_input = self.embed_orth_tokens(orth_dec_input)  # , "o")
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1], orth_dec_input.device)
        orth_output = self.orthography_decoder(
            tgt=orth_dec_input,
            tgt_mask=orth_ar_mask,
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=mixed_encoding,
        )
        orth_token_logits = self.linear_orthography_decoder(orth_output)
        orth_token_logits = orth_token_logits.transpose(1, 2)
        return {"orth": orth_token_logits}
