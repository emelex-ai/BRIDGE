from src.utils.configuration.dataset import DatasetConfig
from src.utils.configuration.model import ModelConfig
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel
import torch.nn as nn
import torch


class BaseModel(ABC, nn.Module):
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig):
        super(BaseModel, self).__init__()
        self.d_model = model_config.d_model
        self.d_embedding = model_config.d_embedding
        self.max_orth_seq_len = dataset_config.max_orth_seq_len
        self.max_phon_seq_len = dataset_config.max_phon_seq_len
        self.nhead = model_config.nhead

        # Initialize embeddings
        self.orthography_embedding = nn.Embedding(len(dataset_config.character_tokenizer), self.d_model)
        self.phonology_embedding = nn.Embedding(len(dataset_config.phonology_tokenizer), self.d_model)
        self.global_embedding = nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model)) / self.d_model**0.5, requires_grad=True
        )

        # Initialize encoders and decoders in child classes
        self.orthography_encoder = None
        self.phonology_encoder = None
        self.orthography_decoder = None
        self.phonology_decoder = None

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        pass

    def generate_triangular_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generates an upper triangular mask."""
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), 1)

    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embeds orthographic tokens."""
        return self.orthography_embedding(tokens)

    def embed_phon_tokens(self, tokens: List[torch.Tensor]) -> torch.Tensor:
        """Embeds phonological tokens."""
        return self.phonology_embedding(tokens)

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
