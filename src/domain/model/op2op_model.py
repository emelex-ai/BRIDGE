from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model, Encoder, Decoder
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class OP2OPModel(Model):

    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(OP2OPModel, self).__init__(model_config, dataset_config)

    def embed_op2op(self, orthography, orthography_padding_mask, phonology, phonology_padding_mask):
        orthography = self.embed_orth_tokens(orthography)
        phonology = self.embed_phon_tokens(phonology)

        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orthography_padding_mask)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phonology_padding_mask)
        # Query = orthography_encoding, Key = phonology_encoding
        gp_encoding = (
            self.gp_multihead_attention(
                orthography_encoding,
                phonology_encoding,
                phonology_encoding,
                key_padding_mask=phonology_padding_mask,
            )[0]
            + orthography_encoding
        )
        gp_encoding = self.gp_layer_norm(gp_encoding)
        # Query = phonology_encoding, Key = orthography_encoding
        pg_encoding = (
            self.pg_multihead_attention(
                phonology_encoding,
                orthography_encoding,
                orthography_encoding,
                key_padding_mask=orthography_padding_mask,
            )[0]
            + phonology_encoding
        )
        pg_encoding = self.pg_layer_norm(pg_encoding)

        # Concatenate outputs of cross-attention modules and add residual connection
        gp_pg = torch.cat((gp_encoding, pg_encoding), dim=1) + torch.cat(
            (orthography_encoding, phonology_encoding), dim=1
        )
        # Concatenate padding masks
        gp_pg_padding_mask = torch.cat((orthography_padding_mask, phonology_padding_mask), dim=-1)

        global_embedding = self.global_embedding.repeat(gp_pg.shape[0], 1, 1)
        gp_pg = torch.cat((global_embedding, gp_pg), dim=1)
        gp_pg_padding_mask = torch.cat(
            (
                torch.zeros(
                    (gp_pg.shape[0], self.d_embedding),
                    device=gp_pg.device,
                    dtype=torch.bool,
                ),
                gp_pg_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(gp_pg, src_key_padding_mask=gp_pg_padding_mask)

        # final_encoding = (self.reduce(mixed_encoding[:, :self.d_embedding]).unsqueeze(-2) + global_embedding)
        # final_encoding = self.reduce_layer_norm(final_encoding)

        # Add a residual connection to the final encoding
        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

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

        mixed_encoding = self.embed_op2op(orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask)
        orth_dec_input = self.embed_orth_tokens(orth_dec_input)
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1], orth_dec_input.device)
        orth_output = self.orthography_decoder(
            tgt=orth_dec_input,
            tgt_mask=orth_ar_mask,
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=mixed_encoding,
        )
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1], phon_dec_input.device)
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=mixed_encoding,
        )
        B, PC, E = phon_output.shape
        orth_token_logits = self.linear_orthography_decoder(orth_output)
        phon_token_logits = self.linear_phonology_decoder(phon_output)
        orth_token_logits = orth_token_logits.transpose(1, 2)
        phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
        return {"orth": orth_token_logits, "phon": phon_token_logits}
