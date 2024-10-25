from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model, Encoder, Decoder
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class O2PModel(Model):
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(O2PModel, self).__init__(model_config, dataset_config)

    def embed_o2p(self, orthography, orthography_padding_mask):
        orthography = self.embed_orth_tokens(orthography)
        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orthography_padding_mask)
        global_embedding = self.global_embedding.repeat(orthography_encoding.shape[0], 1, 1)
        orthography_encoding = torch.cat((global_embedding, orthography_encoding), dim=1)
        orthography_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (orthography_encoding.shape[0], self.d_embedding),
                    device=orthography_encoding.device,
                    dtype=torch.bool,
                ),
                orthography_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(
            orthography_encoding, src_key_padding_mask=orthography_encoding_padding_mask
        )

        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

    def forward(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mixed_encoding = self.embed_o2p(orth_enc_input, orth_enc_pad_mask)
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)  # , "p")
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1], phon_dec_input.device)
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=mixed_encoding,
        )
        B, PC, E = phon_output.shape
        phon_token_logits = self.linear_phonology_decoder(phon_output)
        phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
        return {"phon": phon_token_logits}
