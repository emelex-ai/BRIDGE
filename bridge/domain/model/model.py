from typing import Any, Literal

import torch
import torch.nn as nn
from torchsummary import summary

from bridge.domain.datamodels import BridgeEncoding, GenerationOutput, ModelConfig
from bridge.domain.dataset import BridgeDataset
from bridge.domain.model.decoder import Decoder
from bridge.domain.model.encoder import Encoder
from bridge.domain.model.sliding_window_wrapper import (
    SlidingWindowDecoderWrapper,
    SlidingWindowEncoderWrapper,
)
from bridge.utils import device_manager
from bridge.utils.helper_functions import set_seed


class Model(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        dataset: BridgeDataset,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.dataset = dataset
        self.device = device_manager.device

        if self.model_config.seed:
            set_seed(seed=self.model_config.seed)

        # Get vocabulary sizes from dataset
        self.orthographic_vocabulary_size = dataset.orthographic_vocabulary_size
        self.phonological_vocabulary_size = dataset.phonological_vocabulary_size

        # Hardcoded sequence lengths - will be replaced with dynamic position encoding in the future
        self.max_orth_seq_len = 1024 * 4  # 30
        self.max_phon_seq_len = 1024 * 4  # 30

        # Initialize embeddings and position embeddings
        self.orthography_embedding = nn.Embedding(
            self.orthographic_vocabulary_size, self.model_config.d_model
        )
        # Fixed position embeddings - will be replaced with dynamic position encoding (e.g., RoPE)
        self.orth_position_embedding = nn.Embedding(
            self.max_orth_seq_len, self.model_config.d_model
        )
        self.phonology_embedding = nn.Embedding(
            self.phonological_vocabulary_size, self.model_config.d_model
        )
        # Fixed position embeddings - will be replaced with dynamic position encoding (e.g., RoPE)
        self.phon_position_embedding = nn.Embedding(
            self.max_phon_seq_len, self.model_config.d_model
        )

        self.global_embedding = nn.Parameter(
            torch.randn(
                (1, self.model_config.d_embedding, self.model_config.d_model),
                device=self.device,
            )
            / self.model_config.d_model**0.5,
            requires_grad=True,
        )

        # Create base encoders
        base_orthography_encoder = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_orth_enc_layers,
        )

        base_phonology_encoder = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_phon_enc_layers,
        )

        # Wrap encoders with sliding window capability
        window_size = getattr(self.model_config, "window_size", 61)  # Default ±30 chars
        sliding_window_enabled = getattr(self.model_config, "use_sliding_window", False)
        is_causal = getattr(self.model_config, "is_causal", False)  # New parameter
        max_seq_len = getattr(self.model_config, "max_seq_len", 4096)  # New parameter
        ensure_contiguous = getattr(
            self.model_config, "ensure_contiguous", False
        )  # New parameter

        self.orthography_encoder = SlidingWindowEncoderWrapper(
            base_orthography_encoder,
            window_size=window_size,
            enabled=sliding_window_enabled,
            is_causal=is_causal,  # New parameter
            max_seq_len=max_seq_len,  # New parameter
            ensure_contiguous=ensure_contiguous,  # New parameter
        )

        self.phonology_encoder = SlidingWindowEncoderWrapper(
            base_phonology_encoder,
            window_size=window_size,
            enabled=sliding_window_enabled,
            is_causal=is_causal,  # New parameter
            max_seq_len=max_seq_len,  # New parameter
            ensure_contiguous=ensure_contiguous,  # New parameter
        )

        # Multihead attentions and layer norms
        self.gp_multihead_attention = nn.MultiheadAttention(
            embed_dim=self.model_config.d_model,
            num_heads=self.model_config.nhead,
            batch_first=True,
        )
        self.gp_layer_norm = nn.LayerNorm(self.model_config.d_model)

        self.pg_multihead_attention = nn.MultiheadAttention(
            embed_dim=self.model_config.d_model,
            num_heads=self.model_config.nhead,
            batch_first=True,
        )
        self.pg_layer_norm = nn.LayerNorm(self.model_config.d_model)

        # Wrap transformer mixer with sliding window
        base_transformer_mixer = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_mixing_enc_layers,
        )

        self.transformer_mixer = SlidingWindowEncoderWrapper(
            base_transformer_mixer,
            window_size=window_size,
            enabled=sliding_window_enabled,
            is_causal=is_causal,  # New parameter
            max_seq_len=max_seq_len,  # New parameter
            ensure_contiguous=ensure_contiguous,  # New parameter
        )

        self.reduce = torch.nn.Linear(
            self.model_config.d_model, self.model_config.d_model
        )
        self.reduce_layer_norm = torch.nn.LayerNorm(self.model_config.d_model)

        # Create base decoders
        base_orthography_decoder = Decoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_orth_dec_layers,
        )

        base_phonology_decoder = Decoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_phon_dec_layers,
        )

        # Wrap decoders with sliding window capability
        self.orthography_decoder = SlidingWindowDecoderWrapper(
            base_orthography_decoder,
            window_size=window_size,
            enabled=sliding_window_enabled,
            max_seq_len=max_seq_len,  # New parameter
            ensure_contiguous=ensure_contiguous,  # New parameter
        )

        self.phonology_decoder = SlidingWindowDecoderWrapper(
            base_phonology_decoder,
            window_size=window_size,
            enabled=sliding_window_enabled,
            max_seq_len=max_seq_len,  # New parameter
            ensure_contiguous=ensure_contiguous,  # New parameter
        )

        self.linear_orthography_decoder = nn.Linear(
            self.model_config.d_model, self.orthographic_vocabulary_size
        )

        self.linear_phonology_decoder = nn.Linear(
            self.model_config.d_model,
            2 * (self.phonological_vocabulary_size - 1),
        )

    # Add method to toggle sliding window on/off
    def set_sliding_window(self, enabled: bool) -> None:
        """Enable or disable sliding window attention for all encoders and decoders.

        Args:
            enabled: Whether to enable sliding window attention
        """
        # Update encoder wrappers
        self.orthography_encoder.enabled = enabled
        self.phonology_encoder.enabled = enabled
        self.transformer_mixer.enabled = enabled

        # Update decoder wrappers
        self.orthography_decoder.enabled = enabled
        self.phonology_decoder.enabled = enabled

        print(f"Sliding window attention {'enabled' if enabled else 'disabled'}")

    # Add method to get current sliding window status
    def get_sliding_window_status(self) -> bool:
        """Get current sliding window attention status.

        Returns:
            True if sliding window attention is enabled, False otherwise
        """
        return self.orthography_encoder.enabled  # All wrappers have same status

    # Helper functions
    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return (
            self.orthography_embedding(tokens)
            + self.orth_position_embedding.weight[None, : tokens.shape[1]]
        )

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
        # len(tokens) is the batch size
        output_embedding = torch.zeros(
            (batch_size, max_phon_len, self.model_config.d_model), device=self.device
        )
        for batch_num, batch in enumerate(tokens):
            for indx, tokes in enumerate(batch):
                # Here tokens should be a pytorch tensor of integers.
                # It extracts the indicated rows from self.phonology_embedding
                avg_embedding = self.phonology_embedding(tokes).mean(axis=0)
                # Insert the resulting averaged embedding vector into the
                # output_embedding tensor as a new row
                output_embedding[batch_num, indx, :] = avg_embedding
        return (
            output_embedding
            + self.phon_position_embedding.weight[None, : len(tokens[0])]
        )

    def generate_triangular_mask(self, size: int) -> torch.Tensor:
        return torch.triu(
            torch.ones((size, size), dtype=torch.bool, device=self.device), 1
        )

    def forward(self, task: str, **kwargs) -> dict[str, torch.Tensor]:
        if task == "o2p":
            return self.forward_o2p(**kwargs)
        elif task == "op2op":
            return self.forward_op2op(**kwargs)
        elif task == "p2o":
            return self.forward_p2o(**kwargs)
        elif task == "p2p":
            return self.forward_p2p(**kwargs)
        else:
            raise ValueError("Invalid pathway selected.")

    def embed_o(self, orth_enc_input, orth_enc_pad_mask):
        # Embed the orthographic input tokens
        orthography = self.embed_orth_tokens(
            orth_enc_input
        )  # Shape: (batch_size, seq_len, d_model)
        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orth_enc_pad_mask
        )
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
            (batch_size, 1), device=self.device, dtype=torch.bool
        )  # Shape: (batch_size, 1)

        # Concatenate the zeros padding to the existing padding mask
        orthography_encoding_padding_mask = torch.cat(
            (zeros_padding, orth_enc_pad_mask), dim=1
        )  # Shape: (batch_size, seq_len + 1)
        mixed_encoding = self.transformer_mixer(
            orthography_encoding, src_key_padding_mask=orthography_encoding_padding_mask
        )
        # Extract the global encoding
        final_encoding = (
            mixed_encoding[:, :1, :] + global_embedding
        )  # Shape: (batch_size, 1, d_model)
        return final_encoding

    def forward_o2p(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_dec_input: list[torch.Tensor],
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # Process phonological decoder input
        final_encoding = self.embed_o(orth_enc_input, orth_enc_pad_mask)
        phon_dec_input = self.embed_phon_tokens(
            phon_dec_input
        )  # Shape: (batch_size, phon_seq_len, d_model)
        phon_ar_mask = self.generate_triangular_mask(
            phon_dec_input.shape[1]
        )  # Shape: (phon_seq_len, phon_seq_len)

        # Pass through the phonology decoder
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=final_encoding,
        )

        # Compute the logits
        B, PC, E = phon_output.shape
        phon_token_logits = (
            self.linear_phonology_decoder(phon_output)
            .view(B, PC, 2, -1)
            .transpose(1, 2)
        )
        return {"phon": phon_token_logits}

    def embed_p(
        self, phon_enc_input: list[torch.Tensor], phon_enc_pad_mask: torch.Tensor
    ):
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phon_enc_pad_mask
        )
        global_embedding = self.global_embedding.repeat(
            phonology_encoding.shape[0], 1, 1
        )
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        phonology_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (phonology_encoding.shape[0], self.model_config.d_embedding),
                    device=self.device,
                    dtype=torch.bool,
                ),
                phon_enc_pad_mask,
            ),
            dim=-1,
        )
        mixed_encoding = self.transformer_mixer(
            phonology_encoding, src_key_padding_mask=phonology_encoding_padding_mask
        )
        final_encoding = (
            mixed_encoding[:, : self.model_config.d_embedding] + global_embedding
        )
        return final_encoding

    def forward_p2o(
        self,
        phon_enc_input: list[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
        phon_enc_input: list[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        phon_dec_input: list[torch.Tensor],
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        final_encoding = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        phon_dec_input = self.embed_phon_tokens(phon_dec_input)
        phon_ar_mask = self.generate_triangular_mask(
            phon_dec_input.shape[1]
        )  # Shape: (phon_seq_len, phon_seq_len)
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=final_encoding,
        )
        # Compute the logits
        B, PC, E = phon_output.shape
        phon_token_logits = (
            self.linear_phonology_decoder(phon_output)
            .view(B, PC, 2, -1)
            .transpose(1, 2)
        )
        return {"phon": phon_token_logits}

    def embed_op(
        self, orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
    ):
        orthography = self.embed_orth_tokens(orth_enc_input)
        phonology = self.embed_phon_tokens(phon_enc_input)

        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orth_enc_pad_mask
        )
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phon_enc_pad_mask
        )
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
                    (gp_pg.shape[0], self.model_config.d_embedding),
                    device=self.device,
                    dtype=torch.bool,
                ),
                gp_pg_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(
            gp_pg, src_key_padding_mask=gp_pg_padding_mask
        )

        # Add a residual connection to the final encoding
        final_encoding = (
            mixed_encoding[:, : self.model_config.d_embedding] + global_embedding
        )

        return final_encoding

    def forward_op2op(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: list[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        mixed_encoding = self.embed_op(
            orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
        )
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
        return {"orth": orth_token_logits, "phon": phon_token_logits}

    def ortho_sample(
        self, last_token_probs: torch.Tensor, deterministic: bool
    ) -> torch.Tensor:
        """
        Samples a single orthographic token, either greedily (deterministic) or stochastically.

        Args:
            last_token_probs: Tensor of shape (batch_size, vocab_size) containing
                              the probabilities for the next token.
            deterministic: Whether to sample greedily (True) or from the distribution (False).

        Returns:
            A tensor of shape (batch_size, 1) containing the sampled token IDs.
        """
        if deterministic:
            return last_token_probs.argmax(dim=1, keepdim=True)
        return torch.multinomial(last_token_probs, num_samples=1)

    def phono_sample(
        self, last_token_probs: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Samples phonological features from the model's output distribution.

        last_token_probs is a tensor of shape (batch_size, 2, num_features).
          - The 0th dimension (index = 0) is the probability of the feature being OFF.
          - The 1st dimension (index = 1) is the probability of the feature being ON.

        Args:
            last_token_probs: (B, 2, num_features) distribution for each feature.
            deterministic: Whether to sample greedily (> 0.5 => ON) or via Bernoulli.

        Returns:
            feature_presence: A (B, num_features) binary tensor (0 or 1).
            out_tokens: A list of lists, each inner list contains the active feature indices
                        for that sample. If all features are off, we default to [PAD].

        Sample from phonological decoder output. last_token_probs is a tensor of shape (batch_size, 2, phon_vocab_size),
        where 2 represents the probability dimension and phon_vocab_size is the number of possible phonological vector
        features (including BOS, EOS, UNK, SPC). For the probability dimension (2) the zeroth index is the probability
        of the feature being off, and the first index is the probability of the feature being on.

        For example [0.6, 0.4] -> [feature off, feature on] and in this scenario the feature is off
        """
        # Determine which features are ON.
        if deterministic:
            # Greater than 0.5 probability indicates feature presence
            feature_presence = (last_token_probs[:, 1, :] > 0.5).long()
        else:  # non-deterministic
            feature_presence = torch.bernoulli(last_token_probs[:, 1, :]).long()

        # This returns a tuple of indices ([x_indices], [y_indices]) we need to convert this to a list of lists
        # where each sublist contains the indicies of activate features for each vector in the batch
        batch_indices, feature_indices = torch.where(feature_presence)

        # Group indices by batch item efficiently
        active_features = [[] for _ in range(last_token_probs.size(0))]
        for batch_idx, feature_idx in zip(
            batch_indices.tolist(), feature_indices.tolist()
        ):
            active_features[batch_idx].append(feature_idx)

        # Handle empty features (all OFF) with vectorized operation
        empty_masks = torch.tensor(
            [len(feats) == 0 for feats in active_features],
            device=self.device,
        )
        for i in range(len(active_features)):
            if empty_masks[i]:
                active_features[i] = [self.dataset.tokenizer.phon_pad_id]  # PAD token

        return feature_presence, active_features

    @torch.no_grad()
    def orthography_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_orth_embeddings: torch.Tensor,
        generated_orth_tokens: torch.Tensor,
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ) -> dict[str, Any]:
        """
        Iteratively generates orthographic tokens for all sequences in the batch.

        Args:
            mask: (max_seq_len, max_seq_len) Triangular causal mask for decoder attention
            generated_orth_embeddings: (batch_size, current_seq_len, d_model) Current token embeddings
            generated_orth_tokens: (batch_size, current_seq_len) Tokens generated so far
            prompt_encoding: (batch_size, d_embedding, d_model) Encoder context
            deterministic: Whether to use greedy (True) or stochastic (False) sampling

        Returns:
            Dictionary containing:
                - orth_probs: List of size batch_size, each containing probability distributions
                            for each generation step for that sequence
                - orth_tokens: (batch_size, seq_len) Tensor of generated token sequences
        """
        batch_size = prompt_encoding.size(0)

        # Initialize probability tracking for each sequence in batch
        orth_probs = [[] for _ in range(batch_size)]

        # Add initial probability placeholders for the BOS token
        initial_prob = torch.zeros(
            (batch_size, self.orthographic_vocabulary_size),
            device=self.device,
        )
        initial_prob[:, 0] = 1  # BOS token probability
        for b in range(batch_size):
            orth_probs[b].append(initial_prob[b])

        # Track which sequences have finished generating
        sequence_finished = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )

        for step in range(self.max_orth_seq_len - 1):
            # Check if all sequences have generated an EOS token
            if (generated_orth_tokens == 1).any(dim=1).all():
                break

            step_mask = mask[: step + 1, : step + 1]

            with torch.no_grad():
                # Get decoder output for current step
                orth_output = self.orthography_decoder(
                    generated_orth_embeddings,
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                )

                # Generate logits and probabilities
                linear_output = self.linear_orthography_decoder(orth_output)
                orthography_token_logits = linear_output.transpose(1, 2)
                last_token_logits = orthography_token_logits[:, :, -1]
                last_token_probs = torch.softmax(last_token_logits, dim=1)

                # Store probabilities for each active sequence
                for b in range(batch_size):
                    if not sequence_finished[b]:
                        orth_probs[b].append(last_token_probs[b])

                # Sample next tokens
                new_orthography_tokens = self.ortho_sample(
                    last_token_probs, deterministic
                )

                # Update generated tokens
                generated_orth_tokens = torch.cat(
                    (generated_orth_tokens, new_orthography_tokens), dim=-1
                )

                # Update embeddings for next step
                generated_orth_embeddings = self.embed_orth_tokens(
                    generated_orth_tokens
                )

                # Update which sequences have finished
                sequence_finished = sequence_finished | (
                    new_orthography_tokens == 1
                ).squeeze(-1)

        return {"orth_probs": orth_probs, "orth_tokens": generated_orth_tokens}

    @torch.no_grad()
    def phonology_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_phon_embeddings: torch.Tensor,
        generated_phon_tokens: list[list[torch.Tensor]],
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ) -> dict[str, Any]:
        """Autoregressive generation of phonological features.

        Args:
            mask: Causal mask for decoder attention
            generated_phon_embeddings: Current sequence embeddings
            generated_phon_tokens: List of active feature indices per position
            prompt_encoding: Encoder context (batch_size, 1, hidden_dim)
            deterministic: Sampling strategy flag
        """
        batch_size = prompt_encoding.size(0)

        # Preallocate output tensors for efficiency
        max_len = self.max_phon_seq_len
        phon_probs = [[] for _ in range(batch_size)]
        phon_vecs = [[] for _ in range(batch_size)]

        for step in range(max_len - 1):
            # Get decoder output for current step
            step_mask = mask[: step + 1, : step + 1]
            phon_output = self.phonology_decoder(
                generated_phon_embeddings,
                memory=prompt_encoding,
                tgt_mask=step_mask,
            )

            # Get logits for next position
            B, PC, E = phon_output.shape
            logits = self.linear_phonology_decoder(phon_output)
            logits = logits.view(B, PC, 2, -1).transpose(1, 2)
            last_token_logits = logits[:, :, -1, :]

            # Convert to probabilities
            last_token_probs = torch.softmax(last_token_logits, dim=1)

            # Sample new features
            new_vectors, new_tokens = self.phono_sample(last_token_probs, deterministic)

            # Update tracking for each batch item
            for b in range(batch_size):
                phon_probs[b].append(
                    last_token_probs[b, 1]
                )  # Probability of feature being ON
                phon_vecs[b].append(new_vectors[b])
                generated_phon_tokens[b].append(
                    torch.tensor(new_tokens[b], device=self.device)
                )

            # Update embeddings for next step
            generated_phon_embeddings = self.embed_phon_tokens(generated_phon_tokens)

            # Check for early stopping (if all sequences have hit EOS)
            if all(
                any(self.dataset.tokenizer.phon_eos_id in token for token in tokens)
                for tokens in generated_phon_tokens
            ):
                break

        return {
            "phon_probs": phon_probs,
            "phon_vecs": phon_vecs,
            "phon_tokens": generated_phon_tokens,
        }

    def _generate(
        self,
        pathway: Literal["o2p", "p2o", "op2op", "p2p", "o2o"],
        orth_enc_input: torch.Tensor | None = None,
        orth_enc_pad_mask: torch.Tensor | None = None,
        phon_enc_input: list[list[torch.Tensor]] | None = None,
        phon_enc_pad_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> dict[str, Any]:
        """
        Generates either orthographic tokens or phonological features (or both),
        depending on the chosen pathway.

        Args:
            pathway: One of ["op2op", "o2p", "p2o"].
            orth_enc_input: (batch_size, max_seq_len) input IDs (from an orth encoder).
            orth_enc_pad_mask: (batch_size, max_seq_len) Boolean mask indicating PAD tokens.
            phon_enc_input: (batch_size, max_seq_len) input IDs (from a phon encoder).
            phon_enc_pad_mask: (batch_size, max_seq_len) Boolean mask indicating PAD tokens.
            deterministic: Whether sampling is greedy (True) or stochastic (False).

        Returns:
            A dictionary with keys:
                - "orth_probs", "orth_tokens" (if orthography was generated)
                - "phon_probs", "phon_vecs", "phon_tokens" (if phonology was generated)
                - "global_encoding": The (batch_size, 1, embedding_dim) memory passed to the decoder.

        Once a model has been trained, to perform inference or generate new phonological vectors or orthographic tokens, use this generate function. As
        input, it expects the pathway and the encoded orthography/phonology. Get the encodings from the character_tokenizer.encode or phonology_tokenizer.encode
        routines.

        Parameters:

        Returns:
            The routine returna a dictionary with keys containing the generated data at various levels, along with the global embedding vector.

        Note:
            Only the o2p pathway is currently implemented to support batch processing. Need to add an issue to complete implementation of the
            p2o and op2op batch processing pathways.

        See Also:
            - phonology_decoder_loop
            - phono_sample
            - orthography_decoder_loop
            - ortho_sample
        """
        self._validate_generate_input(
            pathway,
            orth_enc_input,
            orth_enc_pad_mask,
            phon_enc_input,
            phon_enc_pad_mask,
        )

        self.eval()

        # Determine batch size from whichever input is present
        if orth_enc_input is not None:
            batch_size = orth_enc_input.size(0)
        elif phon_enc_input is not None:
            batch_size = len(phon_enc_input)
        else:
            raise ValueError(
                "Neither orthographic nor phonological input provided. "
                "Cannot determine batch size for generation."
            )

        with torch.no_grad():
            # Initialize all outputs to None
            output = {
                "global_encoding": None,
                "orth_probs": None,
                "orth_tokens": None,
                "phon_probs": None,
                "phon_vecs": None,
                "phon_tokens": None,
            }

            if pathway == "op2op":
                output["global_encoding"] = self.embed_op(
                    orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
                )
            elif pathway in ["o2p", "o2o"]:
                output["global_encoding"] = self.embed_o(
                    orth_enc_input, orth_enc_pad_mask
                )
            elif pathway in ["p2o", "p2p"]:
                output["global_encoding"] = self.embed_p(
                    phon_enc_input, phon_enc_pad_mask
                )

            print(f"==> {pathway=}")

            # All these pathways have "2p" meaning we need to run the phonological decoder loop
            if pathway in ["op2op", "o2p", "p2p"]:
                mask = self.generate_triangular_mask(self.max_phon_seq_len)

                generated_phon_tokens = [
                    [
                        torch.tensor(
                            [self.dataset.tokenizer.phon_bos_id],
                            dtype=torch.long,
                            device=self.device,
                        )
                    ]
                    for _ in range(batch_size)
                ]

                generated_phon_embeddings = self.embed_phon_tokens(
                    generated_phon_tokens
                )
                print(f"====> _geneate(), {mask.shape=}")
                generated_phon_results = self.phonology_decoder_loop(
                    mask,
                    generated_phon_embeddings,
                    generated_phon_tokens,
                    output["global_encoding"],
                    deterministic,
                )
                output.update(
                    {
                        "phon_probs": generated_phon_results["phon_probs"],
                        "phon_vecs": generated_phon_results["phon_vecs"],
                        "phon_tokens": generated_phon_results["phon_tokens"],
                    }
                )

            # All these pathways have "2o" meaning we need to run the orthography decoder loop
            if pathway in ["op2op", "p2o", "o2o"]:
                mask = self.generate_triangular_mask(self.max_orth_seq_len)
                generated_orth_tokens = torch.tensor(
                    [[0] for _ in range(batch_size)],
                    dtype=torch.long,
                    device=self.device,
                )
                generated_orth_embeddings = self.embed_orth_tokens(
                    generated_orth_tokens
                )
                generated_orth_results = self.orthography_decoder_loop(
                    mask,
                    generated_orth_embeddings,
                    generated_orth_tokens,
                    output["global_encoding"],
                    deterministic,
                )
                output.update(
                    {
                        "orth_probs": generated_orth_results["orth_probs"],
                        "orth_tokens": generated_orth_results["orth_tokens"],
                    }
                )

            return output

    def generate(
        self,
        encodings: BridgeEncoding,
        pathway: Literal["o2p", "p2o", "op2op", "p2p", "o2o"],
        deterministic: bool = False,
    ) -> GenerationOutput:
        """
        High-level generation interface that works with unified encoding objects.

        This method provides a simplified interface to the model's generation capabilities,
        automatically extracting the appropriate tensors from the encoding object based
        on the selected pathway. It handles all the complexity of routing the correct
        inputs to the underlying generation mechanism.

        Args:
            encodings: A BridgeEncoding object containing orthographic and/or
                    phonological representations, depending on the pathway.
            pathway: The generation pathway to use. Defaults to "o2p" (orthographic
                    to phonological).
            deterministic: Whether to use deterministic (greedy) or stochastic sampling.
                        Defaults to False (stochastic).

        Returns:
            A GenerationOutput object containing the generated sequences and associated
            probability distributions.

        Raises:
            ValueError: If the selected pathway is incompatible with the provided encodings
                    or if any required encoding components are missing.
        """
        # Extract appropriate tensors based on pathway
        orth_enc_input = None
        orth_enc_pad_mask = None
        phon_enc_input = None
        phon_enc_pad_mask = None

        # Handle orthographic inputs for relevant pathways
        if pathway in ["o2p", "o2o", "op2op"]:
            if encodings.orthographic is None:
                raise ValueError(f"Pathway {pathway} requires orthographic encodings")
            orth_enc_input = encodings.orthographic.enc_input_ids
            orth_enc_pad_mask = encodings.orthographic.enc_pad_mask

        # Handle phonological inputs for relevant pathways
        if pathway in ["p2o", "p2p", "op2op"]:
            # For op2op, both are required
            if pathway == "op2op" and encodings.phonological is None:
                raise ValueError(f"Pathway {pathway} requires phonological encodings")

            # For p2o and p2p, we need phonological data
            if pathway in ["p2o", "p2p"] and encodings.phonological is None:
                raise ValueError(f"Pathway {pathway} requires phonological encodings")

            if encodings.phonological is not None:
                phon_enc_input = encodings.phonological.enc_input_ids
                phon_enc_pad_mask = encodings.phonological.enc_pad_mask

        # Call the underlying generate function
        generation_results = self._generate(
            pathway=pathway,
            orth_enc_input=orth_enc_input,
            orth_enc_pad_mask=orth_enc_pad_mask,
            phon_enc_input=phon_enc_input,
            phon_enc_pad_mask=phon_enc_pad_mask,
            deterministic=deterministic,
        )

        # Package the results into our validated GenerationOutput model
        # The model's validators will ensure everything is consistent
        return GenerationOutput(
            global_encoding=generation_results["global_encoding"],
            orth_probs=generation_results.get("orth_probs"),
            orth_tokens=generation_results.get("orth_tokens"),
            phon_probs=generation_results.get("phon_probs"),
            phon_vecs=generation_results.get("phon_vecs"),
            phon_tokens=generation_results.get("phon_tokens"),
        )

    def _validate_generate_input(
        self,
        pathway: str,
        orth_enc_input: torch.Tensor | None,
        orth_enc_pad_mask: torch.Tensor | None,
        phon_enc_input: list[list[torch.Tensor]] | None,
        phon_enc_pad_mask: torch.Tensor | None,
    ) -> None:
        """
        Validates inputs for the generate method based on the selected pathway.

        For the p2o pathway, we need to ensure:
        1. Phonological input is present and properly formatted
        2. Orthographic input is None (since it's not used)
        3. Dimensions and shapes are consistent
        4. Device placement is correct
        """
        if pathway not in ["o2p", "p2o", "op2op", "p2p", "o2o"]:
            raise ValueError(f"Invalid pathway: {pathway}")

        # Add sequence length validation for phonological input
        if phon_enc_input is not None:
            max_phon_len = max(len(seq) for seq in phon_enc_input)
            if max_phon_len > self.max_phon_seq_len:
                raise ValueError(
                    f"Phonological input sequence length {max_phon_len} exceeds "
                    f"maximum allowed length {self.max_phon_seq_len}"
                )

        if pathway == "p2o":
            # Check that orthographic inputs are None
            if orth_enc_input is not None or orth_enc_pad_mask is not None:
                raise ValueError(
                    "p2o pathway expects orthographic inputs (orth_enc_input, orth_enc_pad_mask) "
                    "to be None as they are not used in this pathway."
                )

            # Validate presence of phonological inputs
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    "p2o pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask). "
                    "Received None value(s)."
                )

            # Validate phonological input structure
            if not isinstance(phon_enc_input, list):
                raise TypeError(
                    f"phon_enc_input must be a list of lists of tensors, got {type(phon_enc_input)}"
                )

            if not all(isinstance(batch_item, list) for batch_item in phon_enc_input):
                raise TypeError(
                    "Each item in phon_enc_input must be a list of tensors containing feature indices"
                )

            if not all(
                isinstance(features, torch.Tensor)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise TypeError(
                    "Feature indices in phon_enc_input must be torch.Tensor objects"
                )

            # Validate padding mask
            if not isinstance(phon_enc_pad_mask, torch.Tensor):
                raise TypeError(
                    f"phon_enc_pad_mask must be a torch.Tensor, got {type(phon_enc_pad_mask)}"
                )

            if not phon_enc_pad_mask.dtype == torch.bool:
                raise TypeError(
                    f"phon_enc_pad_mask must be a boolean tensor, got dtype={phon_enc_pad_mask.dtype}"
                )

            # Validate shape consistency
            batch_size = len(phon_enc_input)
            if phon_enc_pad_mask.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: phon_enc_input has {batch_size} items but "
                    f"phon_enc_pad_mask has {phon_enc_pad_mask.size(0)} items"
                )

            # Validate device placement
            if not phon_enc_pad_mask.device == self.device:
                raise ValueError(
                    f"phon_enc_pad_mask must be on device {self.device}, "
                    f"got {phon_enc_pad_mask.device}"
                )

            # Validate feature indices are within vocabulary bounds
            max_feature_idx = self.phonological_vocabulary_size
            if any(
                torch.any(features >= max_feature_idx)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise ValueError(
                    f"Feature indices must be less than vocabulary size "
                    f"({max_feature_idx})"
                )

        # For o2p pathway, validate required inputs
        if pathway == "o2p":
            # Check that required inputs are provided
            if orth_enc_input is None:
                raise ValueError("orth_enc_input is required for o2p pathway")
            if orth_enc_pad_mask is None:
                raise ValueError("orth_enc_pad_mask is required for o2p pathway")

            # Validate input dimensions
            if not isinstance(orth_enc_input, torch.Tensor):
                raise ValueError("orth_enc_input must be a torch.Tensor")
            if orth_enc_input.dim() != 2:
                raise ValueError(
                    "Expected 2D input tensor for orth_enc_input, got shape: "
                    f"{tuple(orth_enc_input.shape)}"
                )

            # Validate input type
            if not (orth_enc_input.dtype in [torch.long, torch.int]):
                raise ValueError(
                    f"orth_enc_input must have dtype torch.long or torch.int, "
                    f"got {orth_enc_input.dtype}"
                )

            # Validate mask dimensions and type
            if not isinstance(orth_enc_pad_mask, torch.Tensor):
                raise ValueError("orth_enc_pad_mask must be a torch.Tensor")
            if orth_enc_pad_mask.dim() != 2:
                raise ValueError("Expected 2D input tensor for orth_enc_pad_mask")
            if not orth_enc_pad_mask.dtype == torch.bool:
                raise ValueError(
                    f"orth_enc_pad_mask must have dtype torch.bool, "
                    f"got {orth_enc_pad_mask.dtype}"
                )

            # Validate matching shapes
            if orth_enc_input.shape != orth_enc_pad_mask.shape:
                raise ValueError(
                    f"Input and mask shapes must match. Got "
                    f"orth_enc_input shape {tuple(orth_enc_input.shape)} and "
                    f"orth_enc_pad_mask shape {tuple(orth_enc_pad_mask.shape)}"
                )

            if pathway == "o2p":
                assert (
                    orth_enc_input is not None
                ), "orth_enc_input is required for o2p pathway."
                assert (
                    orth_enc_pad_mask is not None
                ), "orth_enc_pad_mask is required for o2p pathway."
            elif pathway == "p2o":
                assert (
                    phon_enc_input is not None
                ), "phon_enc_input is required for p2o pathway."
                assert (
                    phon_enc_pad_mask is not None
                ), "phon_enc_pad_mask is required for p2o pathway."
            elif pathway == "op2op":
                assert (
                    orth_enc_input is not None
                ), "orth_enc_input is required for op2op pathway."
                assert (
                    orth_enc_pad_mask is not None
                ), "orth_enc_pad_mask is required for op2op pathway."
                assert (
                    phon_enc_input is not None
                ), "phon_enc_input is required for op2op pathway."
                assert (
                    phon_enc_pad_mask is not None
                ), "phon_enc_pad_mask is required for op2op pathway."
            else:
                raise ValueError("Invalid pathway selected.")

        if pathway == "p2p":
            # Check that orthographic inputs are None
            if orth_enc_input is not None or orth_enc_pad_mask is not None:
                raise ValueError(
                    "p2p pathway expects orthographic inputs (orth_enc_input, orth_enc_pad_mask) "
                    "to be None as they are not used in this pathway."
                )

            # Validate presence of phonological inputs
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    "p2p pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask). "
                    "Received None value(s)."
                )

            # Validate phonological input structure
            if not isinstance(phon_enc_input, list):
                raise TypeError(
                    f"phon_enc_input must be a list of lists of tensors, got {type(phon_enc_input)}"
                )

            if not all(isinstance(batch_item, list) for batch_item in phon_enc_input):
                raise TypeError(
                    "Each item in phon_enc_input must be a list of tensors containing feature indices"
                )

            if not all(
                isinstance(features, torch.Tensor)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise TypeError(
                    "Feature indices in phon_enc_input must be torch.Tensor objects"
                )

            # Validate padding mask
            if not isinstance(phon_enc_pad_mask, torch.Tensor):
                raise TypeError(
                    f"phon_enc_pad_mask must be a torch.Tensor, got {type(phon_enc_pad_mask)}"
                )

            if not phon_enc_pad_mask.dtype == torch.bool:
                raise TypeError(
                    f"phon_enc_pad_mask must be a boolean tensor, got dtype={phon_enc_pad_mask.dtype}"
                )

            # Validate shape consistency
            batch_size = len(phon_enc_input)
            if phon_enc_pad_mask.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: phon_enc_input has {batch_size} items but "
                    f"phon_enc_pad_mask has {phon_enc_pad_mask.size(0)} items"
                )

            # Validate device placement
            if not phon_enc_pad_mask.device == self.device:
                raise ValueError(
                    f"phon_enc_pad_mask must be on device {self.device}, "
                    f"got {phon_enc_pad_mask.device}"
                )

            # Validate feature indices are within vocabulary bounds
            max_feature_idx = self.phonological_vocabulary_size
            if any(
                torch.any(features >= max_feature_idx)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise ValueError(
                    f"Feature indices must be less than vocabulary size "
                    f"({max_feature_idx})"
                )

        if pathway == "o2o":
            # Check that phonological inputs are None
            if phon_enc_input is not None or phon_enc_pad_mask is not None:
                raise ValueError(
                    "o2o pathway expects phonological inputs (phon_enc_input, phon_enc_pad_mask) "
                    "to be None as they are not used in this pathway."
                )

            # Validate presence of orthographic inputs
            if orth_enc_input is None or orth_enc_pad_mask is None:
                raise ValueError(
                    "o2o pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask). "
                    "Received None value(s)."
                )

            # Validate input dimensions and type
            if not isinstance(orth_enc_input, torch.Tensor):
                raise ValueError("orth_enc_input must be a torch.Tensor")

            if orth_enc_input.dim() != 2:
                raise ValueError(
                    f"Expected 2D input tensor for orth_enc_input, got shape: "
                    f"{tuple(orth_enc_input.shape)}"
                )

            # Validate input type
            if not (orth_enc_input.dtype in [torch.long, torch.int]):
                raise ValueError(
                    f"orth_enc_input must have dtype torch.long or torch.int, "
                    f"got {orth_enc_input.dtype}"
                )

            # Validate mask dimensions and type
            if not isinstance(orth_enc_pad_mask, torch.Tensor):
                raise ValueError("orth_enc_pad_mask must be a torch.Tensor")

            if orth_enc_pad_mask.dim() != 2:
                raise ValueError("Expected 2D input tensor for orth_enc_pad_mask")

            if not orth_enc_pad_mask.dtype == torch.bool:
                raise ValueError(
                    f"orth_enc_pad_mask must have dtype torch.bool, "
                    f"got {orth_enc_pad_mask.dtype}"
                )

            # Validate matching shapes
            if orth_enc_input.shape != orth_enc_pad_mask.shape:
                raise ValueError(
                    f"Input and mask shapes must match. Got "
                    f"orth_enc_input shape {tuple(orth_enc_input.shape)} and "
                    f"orth_enc_pad_mask shape {tuple(orth_enc_pad_mask.shape)}"
                )

            # Validate vocabulary bounds
            if torch.any(orth_enc_input >= self.orthographic_vocabulary_size):
                raise ValueError(
                    f"Input tokens must be less than vocabulary size "
                    f"({self.orthographic_vocabulary_size})"
                )

            # Validate device placement
            if orth_enc_input.device != self.device:
                raise ValueError(
                    f"orth_enc_input must be on device {self.device}, "
                    f"got {orth_enc_input.device}"
                )

            if orth_enc_pad_mask.device != self.device:
                raise ValueError(
                    f"orth_enc_pad_mask must be on device {self.device}, "
                    f"got {orth_enc_pad_mask.device}"
                )

        # Specific validation for op2op pathway
        if pathway == "op2op":
            # Verify all inputs are provided
            if orth_enc_input is None or orth_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask)"
                )
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask)"
                )

            # Validate orthographic input structure
            if not isinstance(orth_enc_input, torch.Tensor):
                raise TypeError("orth_enc_input must be a torch.Tensor")
            if orth_enc_input.dim() != 2:
                raise ValueError(
                    f"Expected 2D input tensor for orth_enc_input, got shape: {tuple(orth_enc_input.shape)}"
                )

            # Validate orthographic input type
            if not (orth_enc_input.dtype in [torch.long, torch.int]):
                raise ValueError(
                    f"orth_enc_input must have dtype torch.long or torch.int, got {orth_enc_input.dtype}"
                )

            # Validate orthographic sequence length
            if orth_enc_input.size(1) > self.max_orth_seq_len:
                raise ValueError(
                    f"Orthographic input sequence length {orth_enc_input.size(1)} exceeds "
                    f"maximum allowed length {self.max_orth_seq_len}"
                )

            # Validate orthographic mask dimensions and type
            if not isinstance(orth_enc_pad_mask, torch.Tensor):
                raise TypeError("orth_enc_pad_mask must be a torch.Tensor")
            if orth_enc_pad_mask.dim() != 2:
                raise ValueError("Expected 2D input tensor for orth_enc_pad_mask")
            if not orth_enc_pad_mask.dtype == torch.bool:
                raise ValueError(
                    f"orth_enc_pad_mask must have dtype torch.bool, got {orth_enc_pad_mask.dtype}"
                )

            # Validate matching shapes for orthographic inputs
            if orth_enc_input.shape != orth_enc_pad_mask.shape:
                raise ValueError(
                    f"Input and mask shapes must match. Got orth_enc_input shape "
                    f"{tuple(orth_enc_input.shape)} and orth_enc_pad_mask shape "
                    f"{tuple(orth_enc_pad_mask.shape)}"
                )

            # Validate phonological input structure
            if not isinstance(phon_enc_input, list):
                raise TypeError(
                    f"phon_enc_input must be a list of lists of tensors, got {type(phon_enc_input)}"
                )
            if not all(isinstance(batch_item, list) for batch_item in phon_enc_input):
                raise TypeError("Each item in phon_enc_input must be a list of tensors")
            if not all(
                isinstance(features, torch.Tensor)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise TypeError(
                    "Feature indices in phon_enc_input must be torch.Tensor objects"
                )

            # Validate phonological sequence length
            max_phon_len = max(len(seq) for seq in phon_enc_input)
            if max_phon_len > self.max_phon_seq_len:
                raise ValueError(
                    f"Phonological input sequence length {max_phon_len} exceeds "
                    f"maximum allowed length {self.max_phon_seq_len}"
                )

            # Validate phonological padding mask
            if not isinstance(phon_enc_pad_mask, torch.Tensor):
                raise TypeError(
                    f"phon_enc_pad_mask must be a torch.Tensor, got {type(phon_enc_pad_mask)}"
                )
            if not phon_enc_pad_mask.dtype == torch.bool:
                raise TypeError(
                    f"phon_enc_pad_mask must be a boolean tensor, got dtype={phon_enc_pad_mask.dtype}"
                )

            # Validate batch size consistency
            batch_size = len(phon_enc_input)
            if phon_enc_pad_mask.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: phon_enc_input has {batch_size} items but "
                    f"phon_enc_pad_mask has {phon_enc_pad_mask.size(0)} items"
                )
            if orth_enc_input.size(0) != batch_size:
                raise ValueError(
                    f"Batch size mismatch: orthographic input has {orth_enc_input.size(0)} items "
                    f"but phonological input has {batch_size} items"
                )

            # Validate phonological feature indices are within vocabulary bounds
            max_feature_idx = self.phonological_vocabulary_size
            if any(
                torch.any(features >= max_feature_idx)
                for batch_item in phon_enc_input
                for features in batch_item
            ):
                raise ValueError(
                    f"Feature indices must be less than vocabulary size ({max_feature_idx})"
                )

            # Validate orthographic tokens are within vocabulary bounds
            if torch.any(orth_enc_input >= self.orthographic_vocabulary_size):
                raise ValueError(
                    f"Orthographic tokens must be less than vocabulary size "
                    f"({self.orthographic_vocabulary_size})"
                )


def get_tensor_memory(tensor: torch.Tensor) -> int:
    """Return memory usage of a tensor in bytes."""
    return tensor.element_size() * tensor.numel()


def get_module_memory(model: torch.nn.Module) -> int:
    """Return total memory usage (in bytes) of all parameters, buffers, and persistent tensors."""
    seen = set()
    total = 0

    # Parameters
    for p in model.parameters():
        if id(p) not in seen:
            total += get_tensor_memory(p)
            seen.add(id(p))

    # Buffers
    for b in model.buffers():
        if id(b) not in seen:
            total += get_tensor_memory(b)
            seen.add(id(b))

    # Other persistent tensors (attributes)
    for name, attr in model.__dict__.items():
        if isinstance(attr, torch.Tensor) and id(attr) not in seen:
            total += get_tensor_memory(attr)
            seen.add(id(attr))

    return total


if __name__ == "__main__":
    import torch

    # Example configuration and dataset
    model_config = ModelConfig(
        d_model=1024,
        nhead=16,
        num_phon_enc_layers=16,
        num_orth_enc_layers=16,
        num_mixing_enc_layers=8,
        num_orth_dec_layers=16,
        num_phon_dec_layers=16,
        d_embedding=4,
        seed=42,
    )

    from bridge.domain.model.synthetic_dataset import SyntheticBridgeDataset

    dataset = (
        SyntheticBridgeDataset()
    )  # You need to replace this with actual dataset initialization
    print(f"{dataset=}")

    # Instantiate the model
    model = Model(model_config, dataset)
    print(model)

    # for name, p in model.named_parameters():
    # print(name, p.shape)

    # Calculate the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Calculate the total memory used by the model
    total_memory_bytes = get_module_memory(model)
    print(
        f"Total memory used by the model (parameters, buffers, persistent tensors): {total_memory_bytes / (1024 ** 2):.2f} MB"
    )

    print("==============================")
    # Create sample input tensors for testing
    batch_size, seq_len = 8, 1024 * 4

    # Create dummy orthographic input
    orth_enc_input = torch.randint(
        0, model.orthographic_vocabulary_size, (batch_size, seq_len)
    )
    orth_enc_pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    # Create dummy phonological input (list of tensors)
    phon_enc_input = [
        torch.randint(0, model.phonological_vocabulary_size, (seq_len,))
        for _ in range(batch_size)
    ]
    phon_enc_pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

    # Create dummy decoder inputs
    phon_dec_input = [
        torch.randint(0, model.phonological_vocabulary_size, (seq_len,))
        for _ in range(batch_size)
    ]
    orth_dec_input = torch.randint(
        0, model.orthographic_vocabulary_size, (batch_size, seq_len)
    )

    # Test the model with proper input format
    try:
        with torch.no_grad():
            for i in range(1000):
                print("i= ", i)
                output = model.forward(
                    "op2op",
                    orth_enc_input=orth_enc_input,
                    orth_enc_pad_mask=orth_enc_pad_mask,
                    phon_enc_input=phon_enc_input,
                    phon_enc_pad_mask=phon_enc_pad_mask,
                    phon_dec_input=phon_dec_input,
                    phon_dec_pad_mask=phon_enc_pad_mask,
                    orth_dec_input=orth_dec_input,
                    orth_dec_pad_mask=orth_enc_pad_mask,
                )
        print("Model forward pass successful!")
        print(f"Output keys: {output.keys()}")
        for key, value in output.items():
            print(f"{key} shape: {value.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
