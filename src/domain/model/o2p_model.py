from src.utils.configuration.dataset import DatasetConfig
from src.domain.model import BaseModel, Encoder, Decoder
from src.utils.configuration.model import ModelConfig
from typing import List, Dict, Optional, Union
import torch.nn as nn
import torch


class O2PModel(BaseModel):
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(O2PModel, self).__init__(model_config, dataset_config)

        self.orthography_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_enc_layers
        )
        self.phonology_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_dec_layers
        )
        self.linear_phonology_decoder = nn.Linear(self.d_model, 2 * (len(dataset_config.phonology_tokenizer) - 1))

    def forward(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Encode orthography
        orthography_encoding = self.encode_orthography(orth_enc_input, orth_enc_pad_mask)

        # Add global embedding
        global_embedding = self.global_embedding.repeat(orthography_encoding.shape[0], 1, 1)
        orthography_encoding = torch.cat((global_embedding, orthography_encoding), dim=1)
        mixed_encoding = orthography_encoding[:, : self.d_embedding] + global_embedding

        # Decode phonology
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1], phon_dec_input.device)
        phon_output = self.decode(
            self.phonology_decoder, phon_dec_input, mixed_encoding, phon_ar_mask, phon_dec_pad_mask
        )
        B, PC, E = phon_output.shape
        phon_token_logits = self.linear_phonology_decoder(phon_output)
        phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
        return {"phon": phon_token_logits}
