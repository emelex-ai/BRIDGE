from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model import Model, Encoder, Decoder
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class P2OModel(Model):

    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(P2OModel, self).__init__(model_config, dataset_config)
        self.phonology_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_enc_layers
        )
        self.orthography_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_dec_layers
        )
        self.linear_orthography_decoder = nn.Linear(self.d_model, len(dataset_config.character_tokenizer))

    def forward(
        self,
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Encode phonology
        phonology_encoding = self.encode_phonology(phon_enc_input, phon_enc_pad_mask)

        # Add global embedding
        global_embedding = self.global_embedding.repeat(phonology_encoding.shape[0], 1, 1)
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        mixed_encoding = phonology_encoding[:, : self.d_embedding] + global_embedding

        # Decode orthography
        orth_dec_input = self.embed_orth_tokens(orth_dec_input)
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1], orth_dec_input.device)
        orth_output = self.decode(
            self.orthography_decoder, orth_dec_input, mixed_encoding, orth_ar_mask, orth_dec_pad_mask
        )
        orth_token_logits = self.linear_orthography_decoder(orth_output)
        orth_token_logits = orth_token_logits.transpose(1, 2)
        return {"orth": orth_token_logits}
