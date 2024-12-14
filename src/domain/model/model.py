from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model.encoder import Encoder
from src.domain.model.decoder import Decoder
from torch.nn.utils.rnn import pad_sequence
from src.utils.helper_funtions import set_seed
from typing import Dict, List, Union
from itertools import accumulate
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig, device: torch.device = "cpu") -> None:
        super().__init__()
        self.device = device
        self.d_model: int = model_config.d_model
        self.d_embedding: int = model_config.d_embedding
        self.max_orth_seq_len: int = dataset_config.max_orth_seq_len
        self.max_phon_seq_len: int = dataset_config.max_phon_seq_len
        self.nhead: int = model_config.nhead

        if model_config.seed:
            set_seed(seed=model_config.seed)
        # Initialize embeddings and position embeddings
        self.orthography_embedding = nn.Embedding(dataset_config.orthographic_vocabulary_size, self.d_model)
        self.orth_position_embedding = nn.Embedding(self.max_orth_seq_len, self.d_model)
        self.phonology_embedding = nn.Embedding(dataset_config.phonological_vocabulary_size, self.d_model)
        self.phon_position_embedding = nn.Embedding(self.max_phon_seq_len, self.d_model)

        self.global_embedding = nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model), device=device) / self.d_model**0.5, requires_grad=True
        )
        self.orthography_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_enc_layers
        )

        self.phonology_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_enc_layers
        )

        # Multihead attentions and layer norms
        self.gp_multihead_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.nhead, batch_first=True
        )
        self.gp_layer_norm = nn.LayerNorm(self.d_model)

        self.pg_multihead_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=self.nhead, batch_first=True
        )
        self.pg_layer_norm = nn.LayerNorm(self.d_model)

        self.transformer_mixer = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_mixing_enc_layers
        )

        self.reduce = torch.nn.Linear(self.d_model, self.d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(self.d_model)

        # Decoders and output layers
        self.orthography_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_dec_layers
        )
        self.linear_orthography_decoder = nn.Linear(self.d_model, dataset_config.orthographic_vocabulary_size)

        self.phonology_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_dec_layers
        )
        self.linear_phonology_decoder = nn.Linear(self.d_model, 2 * (dataset_config.phonological_vocabulary_size - 1))

    # Helper functions
    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.orthography_embedding(tokens) + self.orth_position_embedding.weight[None, : tokens.shape[1]]

    def embed_phon_tokens(self, tokens):
        # tokens: list of list of tensors
        try:
            isinstance(tokens, list)
        except:
            raise TypeError(
                f"For phonological vectors, tokens must be a list where each element is \
                    a pytorch tensor of integers (indices), but is type: {type(tokens)}"
            )
        try:
            all(isinstance(token, torch.Tensor) for token in tokens)
        except:
            raise TypeError(
                "For phonological vectors, each element of the list must be \
                    a pytorch tensor of integers (indices)"
            )

        # Here we average the embeddings for each feature in a phonological vector
        # Each row of indices will become of batch once we extract rows from the embedding matrix
        # So the size of the resulting 'output_embedding' tensor should be (batch_size, max_phon_len, d_model)
        batch_size = len(tokens)
        # Every batch should be the same size. If this function is called from the forward routine, then the dataset.encode
        # routine will have already added the necessary padding. If this function is called from the generate routine, then
        # each successive phonological vector (list of active features) will have been generated at the same time. So we can
        # set the max_phon_len to the length of the first batch, since all batches should be the same length.
        max_phon_len = len(tokens[0])
        device = next(self.parameters()).device  # device of weights
        # len(tokens) is the batch size
        output_embedding = torch.zeros((batch_size, max_phon_len, self.d_model), device=device)
        for batch_num, batch in enumerate(tokens):
            for indx, tokes in enumerate(batch):
                # Here tokens should be a pytorch tensor of integers.
                # It extracts the indicated rows from self.phonology_embedding
                avg_embedding = self.phonology_embedding(tokes).mean(axis=0)
                # Insert the resulting averaged embedding vector into the
                # output_embedding tensor as a new row
                output_embedding[batch_num, indx, :] = avg_embedding
        return output_embedding + self.phon_position_embedding.weight[None, : len(tokens[0])]

    def generate_triangular_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=self.device), 1)

    def forward(self, task: str, **kwargs) -> Dict[str, torch.Tensor]:
        if task == "o2p":
            return self.forward_o2p(**kwargs)
        elif task == "op2op":
            return self.forward_op2op(**kwargs)
        elif task == "p2o":
            return self.forward_p2o(**kwargs)
        elif task == "p2p":
            print("P2P Pathway")
            return self.forward_p2p(**kwargs)
        else:
            raise ValueError("Invalid pathway selected.")

    def embed_o2p(self, orth_enc_input, orth_enc_pad_mask):
        # Embed the orthographic input tokens
        orthography = self.embed_orth_tokens(orth_enc_input)  # Shape: (batch_size, seq_len, d_model)
        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orth_enc_pad_mask)
        global_embedding = self.global_embedding.expand(
            orthography_encoding.shape[0], -1, -1
        )  # Shape: (batch_size, 1, d_model)
        # Concatenate the global embedding to the orthography encoding
        orthography_encoding = torch.cat(
            (global_embedding, orthography_encoding), dim=1
        )  # Shape: (batch_size, seq_len + 1, d_model)
        # Create the padding mask for the global embedding
        batch_size = orthography_encoding.shape[0]
        zeros_padding = torch.zeros(
            (batch_size, 1), device=orthography_encoding.device, dtype=torch.bool
        )  # Shape: (batch_size, 1)

        # Concatenate the zeros padding to the existing padding mask
        orthography_encoding_padding_mask = torch.cat(
            (zeros_padding, orth_enc_pad_mask), dim=1
        )  # Shape: (batch_size, seq_len + 1)
        mixed_encoding = self.transformer_mixer(
            orthography_encoding, src_key_padding_mask=orthography_encoding_padding_mask
        )
        # Extract the global encoding
        final_encoding = mixed_encoding[:, :1, :] + global_embedding  # Shape: (batch_size, 1, d_model)
        return final_encoding

    def forward_o2p(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_dec_input: List[torch.Tensor],
        phon_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        # Process phonological decoder input
        final_encoding = self.embed_o2p(orth_enc_input, orth_enc_pad_mask)
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)  # Shape: (batch_size, phon_seq_len, d_model)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1])  # Shape: (phon_seq_len, phon_seq_len)

        # Pass through the phonology decoder
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=final_encoding,
        )

        # Compute the logits
        B, PC, E = phon_output.shape
        phon_token_logits = self.linear_phonology_decoder(phon_output).view(B, PC, 2, -1).transpose(1, 2)
        return {"phon": phon_token_logits}

    def embed_p(self, phon_enc_input: List[torch.Tensor], phon_enc_pad_mask: torch.Tensor):
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phon_enc_pad_mask)
        global_embedding = self.global_embedding.repeat(phonology_encoding.shape[0], 1, 1)
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        phonology_encoding_padding_mask = torch.cat(
            (
                torch.zeros((phonology_encoding.shape[0], self.d_embedding), device=self.device, dtype=torch.bool),
                phon_enc_pad_mask,
            ),
            dim=-1,
        )
        mixed_encoding = self.transformer_mixer(
            phonology_encoding, src_key_padding_mask=phonology_encoding_padding_mask
        )
        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding
        return final_encoding

    def forward_p2o(
        self,
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        final_encoding = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        orth_dec_input = self.embed_orth_tokens(orth_dec_input)
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1])
        orth_output = self.orthography_decoder(
            tgt=orth_dec_input,
            tgt_mask=orth_ar_mask,
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=final_encoding,
        )
        orth_token_logits = self.linear_orthography_decoder(orth_output).transpose(1, 2)
        return {"orth": orth_token_logits}

    def forward_p2p(
        self,
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        phon_dec_input: List[torch.Tensor],
        phon_dec_pad_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        final_encoding = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1])  # Shape: (phon_seq_len, phon_seq_len)
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=final_encoding,
        )
        # Compute the logits
        B, PC, E = phon_output.shape
        phon_token_logits = self.linear_phonology_decoder(phon_output).view(B, PC, 2, -1).transpose(1, 2)
        return {"phon": phon_token_logits}

    def embed_op2op(self, orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask):
        orthography = self.embed_orth_tokens(orth_enc_input)
        phonology = self.embed_phon_tokens(phon_enc_input)

        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orth_enc_pad_mask)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phon_enc_pad_mask)
        # Query = orthography_encoding, Key = phonology_encoding
        gp_encoding = (
            self.gp_multihead_attention(
                orthography_encoding,
                phonology_encoding,
                phonology_encoding,
                key_padding_mask=phon_enc_pad_mask,
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
                key_padding_mask=orth_enc_pad_mask,
            )[0]
            + phonology_encoding
        )
        pg_encoding = self.pg_layer_norm(pg_encoding)

        # Concatenate outputs of cross-attention modules and add residual connection
        gp_pg = torch.cat((gp_encoding, pg_encoding), dim=1) + torch.cat(
            (orthography_encoding, phonology_encoding), dim=1
        )
        # Concatenate padding masks
        gp_pg_padding_mask = torch.cat((orth_enc_pad_mask, phon_enc_pad_mask), dim=-1)

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

    def forward_op2op(
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
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1])
        orth_output = self.orthography_decoder(
            tgt=orth_dec_input,
            tgt_mask=orth_ar_mask,
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=mixed_encoding,
        )
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1])
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
        # print(orth_token_logits)
        return {"orth": orth_token_logits, "phon": phon_token_logits}
