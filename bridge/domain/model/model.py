from collections.abc import Callable
from typing import Literal, TypedDict

import torch
import torch.nn as nn

from bridge.domain.datamodels import BridgeEncoding, GenerationOutput, ModelConfig
from bridge.domain.model.layers import Decoder, Encoder
from bridge.utils import device_manager
from bridge.utils.helper_functions import set_seed

Pathway = Literal["o2p", "p2o", "op2op", "p2p", "o2o"]
PATHWAYS: tuple[Pathway, ...] = ("o2p", "p2o", "op2op", "p2p", "o2o")


class GenerationDict(TypedDict):
    """Internal return shape for `Model._generate`. Fields that don't apply to
    the chosen pathway stay ``None``; ``global_encoding`` is always populated."""

    global_encoding: torch.Tensor
    orth_probs: list[list[torch.Tensor]] | None
    orth_tokens: torch.Tensor | None
    phon_probs: list[list[torch.Tensor]] | None
    phon_vecs: list[list[torch.Tensor]] | None
    phon_tokens: list[list[torch.Tensor]] | None


class Model(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.device = device_manager.device

        if self.model_config.seed:
            set_seed(seed=self.model_config.seed)

        vocab = self.model_config.vocab
        self.orthographic_vocabulary_size = vocab.orth_vocab_size
        self.phonological_vocabulary_size = vocab.phon_vocab_size

        # Hardcoded sequence lengths - will be replaced with dynamic position encoding in the future
        self.max_orth_seq_len = 30
        self.max_phon_seq_len = 30

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
        self.orthography_encoder = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_orth_enc_layers,
        )

        self.phonology_encoder = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_phon_enc_layers,
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

        self.transformer_mixer = Encoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_mixing_enc_layers,
        )

        self.reduce = torch.nn.Linear(self.model_config.d_model, self.model_config.d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(self.model_config.d_model)

        # Decoders and output layers
        self.orthography_decoder = Decoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_orth_dec_layers,
        )
        self.linear_orthography_decoder = nn.Linear(
            self.model_config.d_model, self.orthographic_vocabulary_size
        )

        self.phonology_decoder = Decoder(
            d_model=self.model_config.d_model,
            nhead=self.model_config.nhead,
            num_layers=self.model_config.num_phon_dec_layers,
        )
        self.linear_phonology_decoder = nn.Linear(
            self.model_config.d_model,
            2 * (self.phonological_vocabulary_size - 1),
        )

    # Helper functions
    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return (
            self.orthography_embedding(tokens)
            + self.orth_position_embedding.weight[None, : tokens.shape[1]]
        )

    def embed_phon_tokens(self, tokens) -> torch.Tensor:
        # tokens: list of list of tensors
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
        return output_embedding + self.phon_position_embedding.weight[None, : len(tokens[0])]

    def generate_triangular_mask(self, size: int) -> torch.Tensor:
        return torch.triu(torch.ones((size, size), dtype=torch.bool, device=self.device), 1)

    def forward(self, task: str, **kwargs) -> dict[str, torch.Tensor]:
        pathways: dict[str, Callable[..., dict[str, torch.Tensor]]] = {
            "o2p": self.forward_o2p,
            "op2op": self.forward_op2op,
            "p2o": self.forward_p2o,
            "p2p": self.forward_p2p,
        }
        if task not in pathways:
            raise ValueError("Invalid pathway selected.")
        return pathways[task](**kwargs)

    def _mix_with_global(self, encoding: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Prepend the learned global embedding, run the shared mixer, and return the
        global rows with a residual connection.

        Shared tail of :meth:`embed_o`, :meth:`embed_p` and :meth:`embed_op`.
        Returns shape ``(batch_size, d_embedding, d_model)``.
        """
        d_embedding = self.model_config.d_embedding
        batch_size = encoding.shape[0]

        global_embedding = self.global_embedding.expand(batch_size, -1, -1)
        encoding = torch.cat((global_embedding, encoding), dim=1)
        global_pad_mask = torch.zeros(
            (batch_size, d_embedding), device=self.device, dtype=torch.bool
        )
        pad_mask = torch.cat((global_pad_mask, pad_mask), dim=-1)

        mixed_encoding = self.transformer_mixer(encoding, src_key_padding_mask=pad_mask)
        return mixed_encoding[:, :d_embedding] + global_embedding

    def _decode_orth(
        self,
        memory: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive orthographic decode. Returns (batch, vocab, seq) logits."""
        embeds = self.embed_orth_tokens(orth_dec_input)
        output = self.orthography_decoder(
            tgt=embeds,
            tgt_mask=self.generate_triangular_mask(embeds.shape[1]),
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=memory,
        )
        return self.linear_orthography_decoder(output).transpose(1, 2)

    def _decode_phon(
        self,
        memory: torch.Tensor,
        phon_dec_input: list[list[torch.Tensor]],
        phon_dec_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Autoregressive phonological decode. Returns (batch, 2, seq, features) logits."""
        embeds = self.embed_phon_tokens(phon_dec_input)
        output = self.phonology_decoder(
            tgt=embeds,
            tgt_mask=self.generate_triangular_mask(embeds.shape[1]),
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=memory,
        )
        batch_size, seq_len, _ = output.shape
        return (
            self.linear_phonology_decoder(output).view(batch_size, seq_len, 2, -1).transpose(1, 2)
        )

    def embed_o(self, orth_enc_input, orth_enc_pad_mask):
        """Encode orthography alone into the global representation."""
        orthography = self.embed_orth_tokens(orth_enc_input)
        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orth_enc_pad_mask
        )
        return self._mix_with_global(orthography_encoding, orth_enc_pad_mask)

    def embed_p(self, phon_enc_input: list[list[torch.Tensor]], phon_enc_pad_mask: torch.Tensor):
        """Encode phonology alone into the global representation."""
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phon_enc_pad_mask
        )
        return self._mix_with_global(phonology_encoding, phon_enc_pad_mask)

    def embed_op(self, orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask):
        """Cross-attend orthography and phonology, then mix into the global representation."""
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
        gp_pg_padding_mask = torch.cat((orth_enc_pad_mask, phon_enc_pad_mask), dim=-1)

        return self._mix_with_global(gp_pg, gp_pg_padding_mask)

    def forward_o2p(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_dec_input: list[list[torch.Tensor]],
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_o(orth_enc_input, orth_enc_pad_mask)
        return {"phon": self._decode_phon(memory, phon_dec_input, phon_dec_pad_mask)}

    def forward_p2o(
        self,
        phon_enc_input: list[list[torch.Tensor]],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        return {"orth": self._decode_orth(memory, orth_dec_input, orth_dec_pad_mask)}

    def forward_p2p(
        self,
        phon_enc_input: list[list[torch.Tensor]],
        phon_enc_pad_mask: torch.Tensor,
        phon_dec_input: list[list[torch.Tensor]],
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        return {"phon": self._decode_phon(memory, phon_dec_input, phon_dec_pad_mask)}

    def forward_op2op(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: list[list[torch.Tensor]],
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
        phon_dec_input: list[list[torch.Tensor]],
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_op(orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask)
        return {
            "orth": self._decode_orth(memory, orth_dec_input, orth_dec_pad_mask),
            "phon": self._decode_phon(memory, phon_dec_input, phon_dec_pad_mask),
        }

    def ortho_sample(self, last_token_probs: torch.Tensor, deterministic: bool) -> torch.Tensor:
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

        Args:
            last_token_probs: (B, 2, num_features) distribution for each phonological
                feature (num_features includes BOS, EOS, UNK and SPC). Index 0 of the
                middle dimension is the probability the feature is OFF, index 1 that it
                is ON — e.g. ``[0.6, 0.4]`` means the feature is most likely off.
            deterministic: Whether to sample greedily (> 0.5 => ON) or via Bernoulli.

        Returns:
            feature_presence: A (B, num_features) binary tensor (0 or 1).
            active_features: A list of lists, each inner list contains the active feature
                indices for that sample. If all features are off, we default to [PAD].
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
        active_features: list[list[int]] = [[] for _ in range(last_token_probs.size(0))]
        for batch_idx, feature_idx in zip(
            batch_indices.tolist(), feature_indices.tolist(), strict=False
        ):
            active_features[batch_idx].append(feature_idx)

        # A vector with every feature OFF decodes to [PAD]
        pad_id = self.model_config.vocab.phon_pad_id
        active_features = [feats if feats else [pad_id] for feats in active_features]

        return feature_presence, active_features

    @torch.no_grad()
    def orthography_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_orth_embeddings: torch.Tensor,
        generated_orth_tokens: torch.Tensor,
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ) -> tuple[list[list[torch.Tensor]], torch.Tensor]:
        """
        Iteratively generates orthographic tokens for all sequences in the batch.

        Args:
            mask: (max_seq_len, max_seq_len) Triangular causal mask for decoder attention
            generated_orth_embeddings: (batch_size, current_seq_len, d_model) Current token embeddings
            generated_orth_tokens: (batch_size, current_seq_len) Tokens generated so far
            prompt_encoding: (batch_size, d_embedding, d_model) Encoder context
            deterministic: Whether to use greedy (True) or stochastic (False) sampling

        Returns:
            A tuple of:
                - orth_probs: List of size batch_size, each containing probability distributions
                            for each generation step for that sequence
                - orth_tokens: (batch_size, seq_len) Tensor of generated token sequences
        """
        batch_size = prompt_encoding.size(0)
        bos_id = self.model_config.vocab.orth_bos_id
        eos_id = self.model_config.vocab.orth_eos_id

        # Initialize probability tracking for each sequence in batch
        orth_probs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        # Add initial probability placeholders for the BOS token
        initial_prob = torch.zeros(
            (batch_size, self.orthographic_vocabulary_size),
            device=self.device,
        )
        initial_prob[:, bos_id] = 1
        for b in range(batch_size):
            orth_probs[b].append(initial_prob[b])

        # Track which sequences have finished generating
        sequence_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.max_orth_seq_len - 1):
            # Check if all sequences have generated an EOS token
            if (generated_orth_tokens == eos_id).any(dim=1).all():
                break

            step_mask = mask[: step + 1, : step + 1]

            # Get decoder output for current step
            orth_output = self.orthography_decoder(
                generated_orth_embeddings,
                memory=prompt_encoding,
                tgt_mask=step_mask,
            )

            # Generate logits and probabilities
            orthography_token_logits = self.linear_orthography_decoder(orth_output).transpose(1, 2)
            last_token_logits = orthography_token_logits[:, :, -1]
            last_token_probs = torch.softmax(last_token_logits, dim=1)

            # Store probabilities for each active sequence
            for b in range(batch_size):
                if not sequence_finished[b]:
                    orth_probs[b].append(last_token_probs[b])

            # Sample next tokens
            new_orthography_tokens = self.ortho_sample(last_token_probs, deterministic)

            # Update generated tokens and the embeddings fed to the next step
            generated_orth_tokens = torch.cat(
                (generated_orth_tokens, new_orthography_tokens), dim=-1
            )
            generated_orth_embeddings = self.embed_orth_tokens(generated_orth_tokens)

            # Update which sequences have finished
            sequence_finished = sequence_finished | (new_orthography_tokens == eos_id).squeeze(-1)

        return orth_probs, generated_orth_tokens

    @torch.no_grad()
    def phonology_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_phon_embeddings: torch.Tensor,
        generated_phon_tokens: list[list[torch.Tensor]],
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """Autoregressive generation of phonological features.

        Args:
            mask: Causal mask for decoder attention
            generated_phon_embeddings: Current sequence embeddings
            generated_phon_tokens: List of active feature indices per position
            prompt_encoding: Encoder context (batch_size, 1, hidden_dim)
            deterministic: Sampling strategy flag
        """
        batch_size = prompt_encoding.size(0)
        phon_eos_id = self.model_config.vocab.phon_eos_id

        phon_probs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        phon_vecs: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]

        for step in range(self.max_phon_seq_len - 1):
            # Get decoder output for current step
            step_mask = mask[: step + 1, : step + 1]
            phon_output = self.phonology_decoder(
                generated_phon_embeddings,
                memory=prompt_encoding,
                tgt_mask=step_mask,
            )

            # Get logits for next position, then convert to probabilities
            batch, seq_len, _ = phon_output.shape
            logits = self.linear_phonology_decoder(phon_output).view(batch, seq_len, 2, -1)
            last_token_logits = logits.transpose(1, 2)[:, :, -1, :]
            last_token_probs = torch.softmax(last_token_logits, dim=1)

            # Sample new features
            new_vectors, new_tokens = self.phono_sample(last_token_probs, deterministic)

            # Update tracking for each batch item
            for b in range(batch_size):
                phon_probs[b].append(last_token_probs[b, 1])  # Probability of feature being ON
                phon_vecs[b].append(new_vectors[b])
                generated_phon_tokens[b].append(torch.tensor(new_tokens[b], device=self.device))

            # Update embeddings for next step
            generated_phon_embeddings = self.embed_phon_tokens(generated_phon_tokens)

            # Check for early stopping (if all sequences have hit EOS)
            if all(
                any(phon_eos_id in token for token in tokens) for tokens in generated_phon_tokens
            ):
                break

        return phon_probs, phon_vecs, generated_phon_tokens

    def _generate(
        self,
        pathway: Pathway,
        orth_enc_input: torch.Tensor | None = None,
        orth_enc_pad_mask: torch.Tensor | None = None,
        phon_enc_input: list[list[torch.Tensor]] | None = None,
        phon_enc_pad_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> GenerationDict:
        """
        Generates either orthographic tokens or phonological features (or both),
        depending on the chosen pathway.

        Encode the inputs with ``BridgeTokenizer.encode`` (or the individual character /
        phoneme tokenizers) before calling this; :meth:`generate` is the friendlier
        entry point that unpacks a :class:`BridgeEncoding` for you.

        Args:
            pathway: One of ``PATHWAYS``.
            orth_enc_input: (batch_size, max_seq_len) input IDs (from an orth encoder).
            orth_enc_pad_mask: (batch_size, max_seq_len) Boolean mask indicating PAD tokens.
            phon_enc_input: (batch_size, max_seq_len) input IDs (from a phon encoder).
            phon_enc_pad_mask: (batch_size, max_seq_len) Boolean mask indicating PAD tokens.
            deterministic: Whether sampling is greedy (True) or stochastic (False).

        Returns:
            A ``GenerationDict``. ``global_encoding`` — the (batch_size, d_embedding,
            d_model) memory passed to the decoders — is always populated; the
            orthographic and phonological fields are filled in only for the pathways
            that generate them, and stay ``None`` otherwise.

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

        # Validation above guarantees the pathway's own modality is present.
        batch_size = (
            orth_enc_input.size(0) if orth_enc_input is not None else len(phon_enc_input)  # type: ignore[arg-type]
        )

        with torch.no_grad():
            # `_validate_generate_input` has already enforced which inputs are
            # non-None for each pathway; the asserts below re-state those
            # invariants so the type checker can narrow.
            if pathway == "op2op":
                assert orth_enc_input is not None and orth_enc_pad_mask is not None
                assert phon_enc_input is not None and phon_enc_pad_mask is not None
                global_encoding = self.embed_op(
                    orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
                )
            elif pathway in ["o2p", "o2o"]:
                assert orth_enc_input is not None and orth_enc_pad_mask is not None
                global_encoding = self.embed_o(orth_enc_input, orth_enc_pad_mask)
            else:  # "p2o", "p2p"
                assert phon_enc_input is not None and phon_enc_pad_mask is not None
                global_encoding = self.embed_p(phon_enc_input, phon_enc_pad_mask)

            output: GenerationDict = {
                "global_encoding": global_encoding,
                "orth_probs": None,
                "orth_tokens": None,
                "phon_probs": None,
                "phon_vecs": None,
                "phon_tokens": None,
            }

            # All these pathways have "2p" meaning we need to run the phonological decoder loop
            if pathway in ["op2op", "o2p", "p2p"]:
                mask = self.generate_triangular_mask(self.max_phon_seq_len)

                generated_phon_tokens = [
                    [
                        torch.tensor(
                            [self.model_config.vocab.phon_bos_id],
                            dtype=torch.long,
                            device=self.device,
                        )
                    ]
                    for _ in range(batch_size)
                ]

                generated_phon_embeddings = self.embed_phon_tokens(generated_phon_tokens)
                phon_probs, phon_vecs, phon_tokens = self.phonology_decoder_loop(
                    mask,
                    generated_phon_embeddings,
                    generated_phon_tokens,
                    global_encoding,
                    deterministic,
                )
                output["phon_probs"] = phon_probs
                output["phon_vecs"] = phon_vecs
                output["phon_tokens"] = phon_tokens

            # All these pathways have "2o" meaning we need to run the orthography decoder loop
            if pathway in ["op2op", "p2o", "o2o"]:
                mask = self.generate_triangular_mask(self.max_orth_seq_len)
                generated_orth_tokens = torch.full(
                    (batch_size, 1),
                    self.model_config.vocab.orth_bos_id,
                    dtype=torch.long,
                    device=self.device,
                )
                generated_orth_embeddings = self.embed_orth_tokens(generated_orth_tokens)
                orth_probs, orth_tokens = self.orthography_decoder_loop(
                    mask,
                    generated_orth_embeddings,
                    generated_orth_tokens,
                    global_encoding,
                    deterministic,
                )
                output["orth_probs"] = orth_probs
                output["orth_tokens"] = orth_tokens

            return output

    def generate(
        self,
        encodings: BridgeEncoding,
        pathway: Pathway,
        deterministic: bool = False,
    ) -> GenerationOutput:
        """
        High-level generation interface that works with unified encoding objects.

        Unpacks the encoder-side tensors that ``pathway`` consumes out of ``encodings``
        and hands them to :meth:`_generate`, then wraps the result in a validated
        :class:`GenerationOutput`.

        Args:
            encodings: A BridgeEncoding object containing orthographic and
                    phonological representations.
            pathway: The generation pathway to use.
            deterministic: Whether to use deterministic (greedy) or stochastic sampling.
                        Defaults to False (stochastic).

        Returns:
            A GenerationOutput object containing the generated sequences and associated
            probability distributions.

        Raises:
            ValueError: If the selected pathway is incompatible with the provided encodings.
        """
        uses_orth = pathway in ("o2p", "o2o", "op2op")
        uses_phon = pathway in ("p2o", "p2p", "op2op")

        generation_results = self._generate(
            pathway=pathway,
            orth_enc_input=encodings.orthographic.enc_input_ids if uses_orth else None,
            orth_enc_pad_mask=encodings.orthographic.enc_pad_mask if uses_orth else None,
            phon_enc_input=encodings.phonological.enc_input_ids if uses_phon else None,
            phon_enc_pad_mask=encodings.phonological.enc_pad_mask if uses_phon else None,
            deterministic=deterministic,
        )

        # GenerationDict's keys are exactly GenerationOutput's fields; the pydantic
        # validators check cross-field consistency.
        return GenerationOutput(**generation_results)

    def _validate_orth_inputs(
        self,
        orth_enc_input: torch.Tensor | None,
        orth_enc_pad_mask: torch.Tensor | None,
        *,
        tensor_exc: type[Exception] = ValueError,
        check_seq_len: bool = False,
    ) -> None:
        """Structural checks shared by every pathway that consumes orthography.

        ``tensor_exc`` selects the exception raised when an argument is not a tensor
        at all: ``op2op`` reports that as a ``TypeError``, the single-modality
        pathways as a ``ValueError``.
        """
        if not isinstance(orth_enc_input, torch.Tensor):
            raise tensor_exc("orth_enc_input must be a torch.Tensor")
        if orth_enc_input.dim() != 2:
            raise ValueError(
                "Expected 2D input tensor for orth_enc_input, got shape: "
                f"{tuple(orth_enc_input.shape)}"
            )
        if orth_enc_input.dtype not in [torch.long, torch.int]:
            raise ValueError(
                f"orth_enc_input must have dtype torch.long or torch.int, "
                f"got {orth_enc_input.dtype}"
            )
        if check_seq_len and orth_enc_input.size(1) > self.max_orth_seq_len:
            raise ValueError(
                f"Orthographic input sequence length {orth_enc_input.size(1)} exceeds "
                f"maximum allowed length {self.max_orth_seq_len}"
            )

        if not isinstance(orth_enc_pad_mask, torch.Tensor):
            raise tensor_exc("orth_enc_pad_mask must be a torch.Tensor")
        if orth_enc_pad_mask.dim() != 2:
            raise ValueError("Expected 2D input tensor for orth_enc_pad_mask")
        if orth_enc_pad_mask.dtype != torch.bool:
            raise ValueError(
                f"orth_enc_pad_mask must have dtype torch.bool, got {orth_enc_pad_mask.dtype}"
            )

        if orth_enc_input.shape != orth_enc_pad_mask.shape:
            raise ValueError(
                f"Input and mask shapes must match. Got "
                f"orth_enc_input shape {tuple(orth_enc_input.shape)} and "
                f"orth_enc_pad_mask shape {tuple(orth_enc_pad_mask.shape)}"
            )

    def _validate_phon_inputs(
        self,
        phon_enc_input: list[list[torch.Tensor]] | None,
        phon_enc_pad_mask: torch.Tensor | None,
    ) -> None:
        """Structural checks shared by every pathway that consumes phonology."""
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
            raise TypeError("Feature indices in phon_enc_input must be torch.Tensor objects")

        if not isinstance(phon_enc_pad_mask, torch.Tensor):
            raise TypeError(
                f"phon_enc_pad_mask must be a torch.Tensor, got {type(phon_enc_pad_mask)}"
            )
        if phon_enc_pad_mask.dtype != torch.bool:
            raise TypeError(
                f"phon_enc_pad_mask must be a boolean tensor, got dtype={phon_enc_pad_mask.dtype}"
            )

        batch_size = len(phon_enc_input)
        if phon_enc_pad_mask.size(0) != batch_size:
            raise ValueError(
                f"Batch size mismatch: phon_enc_input has {batch_size} items but "
                f"phon_enc_pad_mask has {phon_enc_pad_mask.size(0)} items"
            )

    def _validate_phon_bounds(self, phon_enc_input: list[list[torch.Tensor]]) -> None:
        """Check phonological feature indices fit the phonological vocabulary."""
        max_feature_idx = self.phonological_vocabulary_size
        if any(
            torch.any(features >= max_feature_idx)
            for batch_item in phon_enc_input
            for features in batch_item
        ):
            raise ValueError(
                f"Feature indices must be less than vocabulary size ({max_feature_idx})"
            )

    def _validate_orth_bounds(self, orth_enc_input: torch.Tensor, label: str) -> None:
        """Check orthographic token ids fit the orthographic vocabulary."""
        if torch.any(orth_enc_input >= self.orthographic_vocabulary_size):
            raise ValueError(
                f"{label} must be less than vocabulary size ({self.orthographic_vocabulary_size})"
            )

    def _validate_device(self, **tensors: torch.Tensor) -> None:
        """Check the named tensors all live on the model's device."""
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on device {self.device}, got {tensor.device}")

    def _validate_generate_input(
        self,
        pathway: str,
        orth_enc_input: torch.Tensor | None,
        orth_enc_pad_mask: torch.Tensor | None,
        phon_enc_input: list[list[torch.Tensor]] | None,
        phon_enc_pad_mask: torch.Tensor | None,
    ) -> None:
        """Validate the encoder inputs against the selected pathway.

        Each branch below states only what is specific to its pathway — which
        modalities are required, which must be absent, and which bounds/device
        checks apply. The per-modality structural checks are shared via
        :meth:`_validate_orth_inputs` and :meth:`_validate_phon_inputs`.
        """
        if pathway not in PATHWAYS:
            raise ValueError(f"Invalid pathway: {pathway}")

        if phon_enc_input is not None:
            max_phon_len = max(len(seq) for seq in phon_enc_input)
            if max_phon_len > self.max_phon_seq_len:
                raise ValueError(
                    f"Phonological input sequence length {max_phon_len} exceeds "
                    f"maximum allowed length {self.max_phon_seq_len}"
                )

        if pathway in ("p2o", "p2p"):
            if orth_enc_input is not None or orth_enc_pad_mask is not None:
                raise ValueError(
                    f"{pathway} pathway expects orthographic inputs (orth_enc_input, "
                    "orth_enc_pad_mask) to be None as they are not used in this pathway."
                )
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    f"{pathway} pathway requires phonological inputs (phon_enc_input, "
                    "phon_enc_pad_mask). Received None value(s)."
                )
            self._validate_phon_inputs(phon_enc_input, phon_enc_pad_mask)
            self._validate_device(phon_enc_pad_mask=phon_enc_pad_mask)
            self._validate_phon_bounds(phon_enc_input)

        elif pathway == "o2p":
            if orth_enc_input is None:
                raise ValueError("orth_enc_input is required for o2p pathway")
            if orth_enc_pad_mask is None:
                raise ValueError("orth_enc_pad_mask is required for o2p pathway")
            self._validate_orth_inputs(orth_enc_input, orth_enc_pad_mask)

        elif pathway == "o2o":
            if phon_enc_input is not None or phon_enc_pad_mask is not None:
                raise ValueError(
                    "o2o pathway expects phonological inputs (phon_enc_input, phon_enc_pad_mask) "
                    "to be None as they are not used in this pathway."
                )
            if orth_enc_input is None or orth_enc_pad_mask is None:
                raise ValueError(
                    "o2o pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask). "
                    "Received None value(s)."
                )
            self._validate_orth_inputs(orth_enc_input, orth_enc_pad_mask)
            self._validate_orth_bounds(orth_enc_input, "Input tokens")
            self._validate_device(
                orth_enc_input=orth_enc_input, orth_enc_pad_mask=orth_enc_pad_mask
            )

        else:  # op2op — the only pathway consuming both modalities
            if orth_enc_input is None or orth_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask)"
                )
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask)"
                )
            self._validate_orth_inputs(
                orth_enc_input,
                orth_enc_pad_mask,
                tensor_exc=TypeError,
                check_seq_len=True,
            )
            self._validate_phon_inputs(phon_enc_input, phon_enc_pad_mask)
            if orth_enc_input.size(0) != len(phon_enc_input):
                raise ValueError(
                    f"Batch size mismatch: orthographic input has {orth_enc_input.size(0)} items "
                    f"but phonological input has {len(phon_enc_input)} items"
                )
            self._validate_phon_bounds(phon_enc_input)
            self._validate_orth_bounds(orth_enc_input, "Orthographic tokens")
