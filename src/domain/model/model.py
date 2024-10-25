from src.domain.datamodels import DatasetConfig, ModelConfig
from typing import List, Dict, Optional, Union
from src.domain.model.encoder import Encoder
from src.domain.model.decoder import Decoder
from abc import ABC, abstractmethod
import torch.nn as nn
import torch


class Model(ABC, nn.Module):
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super().__init__()
        self.d_model = model_config.d_model
        self.d_embedding = model_config.d_embedding
        self.max_orth_seq_len = dataset_config.max_orth_seq_len
        self.max_phon_seq_len = dataset_config.max_phon_seq_len
        self.nhead = model_config.nhead

        # Initialize embeddings
        self.orthography_embedding = nn.Embedding(dataset_config.orthographic_vocabulary_size, self.d_model)
        self.orth_position_embedding = torch.nn.Embedding(self.max_orth_seq_len, self.d_model)

        self.phonology_embedding = nn.Embedding(dataset_config.phonological_vocabulary_size, self.d_model)
        self.phon_position_embedding = torch.nn.Embedding(self.max_phon_seq_len, self.d_model)

        self.global_embedding = nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model)) / self.d_model**0.5, requires_grad=True
        )
        # Initial, encoding segment of our ConnTextUL model:
        # Instance of our Encoder module (defined above), for encoding orthography
        self.orthography_encoder = Encoder(
            d_model=self.d_model, nhead=model_config.nhead, num_layers=model_config.num_orth_enc_layers
        )
        # Instance of our Encoder module (defined above), for encoding phonology
        self.phonology_encoder = Encoder(
            d_model=self.d_model, nhead=model_config.nhead, num_layers=model_config.num_phon_enc_layers
        )

        # Criss-crossing orthography/phonology cross-attenion segment of ConnTextUL model
        self.gp_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=model_config.nhead, batch_first=True
        )
        self.gp_layer_norm = torch.nn.LayerNorm(self.d_model)
        self.pg_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=model_config.nhead, batch_first=True
        )
        self.pg_layer_norm = torch.nn.LayerNorm(self.d_model)

        # Segment of ConnTextUL model that mixes orthography/phonology representation
        self.transformer_mixer = Encoder(
            d_model=self.d_model,
            nhead=model_config.nhead,
            num_layers=model_config.num_mixing_enc_layers,
        )
        self.reduce = torch.nn.Linear(self.d_model, self.d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(self.d_model)

        self.orthography_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_dec_layers
        )
        self.phonology_decoder = Decoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_dec_layers
        )

        self.linear_orthography_decoder = nn.Linear(self.d_model, dataset_config.orthographic_vocabulary_size)
        self.linear_phonology_decoder = nn.Linear(self.d_model, 2 * (dataset_config.phonological_vocabulary_size - 1))

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def generate_triangular_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generates an upper triangular mask."""
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), 1)

    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds orthographic tokens."""
        assert isinstance(
            tokens, torch.Tensor
        ), "For orthographic embeddings, tokens must be a pytorch tensor of integers (indices of orthography_embedding)"
        assert (
            tokens.dtype == torch.long or tokens.dtype == torch.int
        ), f"Input tensor to Embedding must be type int or long but is {tokens.dtype}"
        return self.orthography_embedding(tokens) + self.orth_position_embedding.weight[None, : tokens.shape[1]]

    def embed_phon_tokens(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Embeds phonological tokens."""
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

    def encode_orthography(self, orthography: torch.Tensor, orthography_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encodes orthography using the shared orthography encoder."""
        orthography = self.embed_orth_tokens(orthography)
        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orthography_padding_mask)
        return orthography_encoding

    def encode_phonology(self, phonology: List[torch.Tensor], phonology_padding_mask: torch.Tensor) -> torch.Tensor:
        """Encodes phonology using the shared phonology encoder."""
        phonology = self.embed_phon_tokens(phonology)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phonology_padding_mask)
        return phonology_encoding

    def decode(
        self,
        decoder: nn.Module,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes the target sequence using a given decoder."""
        return decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
