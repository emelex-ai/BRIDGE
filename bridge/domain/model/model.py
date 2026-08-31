from collections.abc import Callable
from typing import Literal, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from bridge.core.phonreps import load_phoneme_table, row_normalize
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

        # `is not None`, not truthiness: 0 is a perfectly good seed and a falsy one, so
        # `if seed:` left `seed=0` unseeded with no warning. An ensemble built over
        # `range(n)` then had a member 0 that did not reproduce while every other member
        # did, and the discrepancy looked like ordinary training noise.
        if self.model_config.seed is not None:
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
        # Row-normalized phoneme -> feature matrix. `phon_feature_matrix @ phonology_embedding
        # .weight` is every phoneme's mean-of-features embedding in one small matmul, which
        # turns embedding a batch into a table lookup. Not a parameter and not persistent:
        # it is derived from phonreps.csv, so it stays out of state_dict and existing
        # checkpoints keep loading with strict=True.
        # Annotated before register_buffer so the attribute types as Tensor, not
        # `Tensor | Module` (nn.Module.__getattr__'s union).
        self.phon_feature_matrix: torch.Tensor
        phoneme_table = load_phoneme_table(device=self.device)
        if phoneme_table.vocab_size != self.phonological_vocabulary_size:
            raise ValueError(
                f"vocab.phon_vocab_size is {self.phonological_vocabulary_size}, but the "
                f"phoneme feature table in phonreps.csv defines "
                f"{phoneme_table.vocab_size} features. The phonological vocabulary is not "
                f"free; it is the feature table plus its special tokens. Build the "
                f"VocabSpec with VocabSpec.from_tokenizer(...) rather than by hand."
            )
        # The special-token ids index the same feature space the table defines, and
        # generation indexes with them directly. A wrong-but-in-range id produces silently
        # wrong output rather than an error, so check them here alongside the size.
        wrong = {
            name: (getattr(vocab, name), phoneme_table.feature_of(token))
            for name, token in (
                ("phon_pad_id", "[PAD]"),
                ("phon_bos_id", "[BOS]"),
                ("phon_eos_id", "[EOS]"),
            )
            if getattr(vocab, name) != phoneme_table.feature_of(token)
        }
        if wrong:
            detail = ", ".join(f"{n}={got} (table says {want})" for n, (got, want) in wrong.items())
            raise ValueError(
                f"VocabSpec special-token ids disagree with phonreps.csv: {detail}. Build the "
                f"VocabSpec with VocabSpec.from_tokenizer(...) rather than by hand."
            )
        self.register_buffer(
            "phon_feature_matrix", row_normalize(phoneme_table.multihot), persistent=False
        )
        # Recorded so a checkpoint can be compared against the table it was trained on.
        self.phon_table_fingerprint = phoneme_table.fingerprint
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
    @staticmethod
    def _positions(table: nn.Embedding, seq_len: int, offset: int) -> torch.Tensor:
        """``seq_len`` position embeddings from ``table``, starting at ``offset``.

        Every embedder takes an ``offset`` so it can embed a slice of a sequence rather
        than only a whole one, which is what lets the generation loops append a single
        row per step instead of re-embedding their whole prefix.
        """
        if offset + seq_len > table.num_embeddings:
            raise ValueError(
                f"Positions {offset}..{offset + seq_len} exceed the {table.num_embeddings}-slot "
                f"position table; the sequence is longer than this model supports."
            )
        return table.weight[offset : offset + seq_len][None]

    def embed_orth_tokens(self, tokens: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        return self.orthography_embedding(tokens) + self._positions(
            self.orth_position_embedding, tokens.shape[1], position_offset
        )

    def embed_phon_tokens(self, rows: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Embed ``(batch, seq)`` phoneme row ids as ``(batch, seq, d_model)``.

        A phoneme's embedding is the mean of its active features' embeddings. That is linear
        in the embedding weight, so the mean for *every* phoneme is one small matmul over the
        91-row feature matrix, independent of batch size, after which embedding a batch is
        a table lookup. The table is rebuilt each call because the weight trains.

        Featureless phonemes (``phonreps.csv`` contains one, ``'_'``) embed to the zero
        vector rather than NaN.
        """
        phoneme_embeddings = self.phon_feature_matrix @ self.phonology_embedding.weight
        return F.embedding(rows, phoneme_embeddings) + self._positions(
            self.phon_position_embedding, rows.shape[1], position_offset
        )

    def embed_phon_vectors(self, multihot: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        """Embed ``(batch, seq, features)`` binary feature vectors as ``(batch, seq, d_model)``.

        The generation counterpart of :meth:`embed_phon_tokens`: sampled feature vectors need
        not correspond to any real phoneme, so no table row exists for them and the mean has
        to be taken over the vector itself.
        """
        normalized = row_normalize(multihot.to(self.phonology_embedding.weight.dtype))
        averaged = normalized @ self.phonology_embedding.weight
        return averaged + self._positions(
            self.phon_position_embedding, multihot.shape[1], position_offset
        )

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
        phon_dec_input: torch.Tensor,
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

    def embed_o(
        self, orth_enc_input: torch.Tensor, orth_enc_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode orthography alone into the global representation."""
        orthography = self.embed_orth_tokens(orth_enc_input)
        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orth_enc_pad_mask
        )
        return self._mix_with_global(orthography_encoding, orth_enc_pad_mask)

    def embed_p(
        self, phon_enc_input: torch.Tensor, phon_enc_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode phonology alone into the global representation."""
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phon_enc_pad_mask
        )
        return self._mix_with_global(phonology_encoding, phon_enc_pad_mask)

    def embed_op(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: torch.Tensor,
        phon_enc_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
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
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_o(orth_enc_input, orth_enc_pad_mask)
        return {"phon": self._decode_phon(memory, phon_dec_input, phon_dec_pad_mask)}

    def forward_p2o(
        self,
        phon_enc_input: torch.Tensor,
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        return {"orth": self._decode_orth(memory, orth_dec_input, orth_dec_pad_mask)}

    def forward_p2p(
        self,
        phon_enc_input: torch.Tensor,
        phon_enc_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
        phon_dec_pad_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self.embed_p(phon_enc_input, phon_enc_pad_mask)
        return {"phon": self._decode_phon(memory, phon_dec_input, phon_dec_pad_mask)}

    def forward_op2op(
        self,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: torch.Tensor,
        phon_enc_pad_mask: torch.Tensor,
        orth_dec_input: torch.Tensor,
        orth_dec_pad_mask: torch.Tensor,
        phon_dec_input: torch.Tensor,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples phonological features from the model's output distribution.

        Args:
            last_token_probs: (B, 2, num_features) distribution for each phonological
                feature (num_features includes BOS, EOS, UNK and SPC). Index 0 of the
                middle dimension is the probability the feature is OFF, index 1 that it
                is ON. So ``[0.6, 0.4]`` means the feature is most likely off.
            deterministic: Whether to sample greedily (> 0.5 => ON) or via Bernoulli.

        Returns:
            feature_presence: A (B, num_predicted) binary tensor (0 or 1), exactly as
                sampled, so an all-off row stays all-off. This is what ``phon_vecs``
                records.
            embedding_input: A (B, phon_vocab_size) binary tensor: ``feature_presence``
                widened by one column and, for all-off rows only, [PAD] switched on. This
                is what gets embedded and what ``phon_tokens`` is decoded from.

        The two differ deliberately, and the asymmetry predates this implementation: an
        all-off vector is *reported* as all-off but *decodes* to [PAD]. Collapsing them
        into one tensor would silently change generation output.

        [PAD] is never predicted (the decoder emits ``phon_vocab_size - 1`` features),
        so the embedding input is one column wider than the sampled presence.
        """
        if deterministic:
            # Greater than 0.5 probability indicates feature presence
            feature_presence = (last_token_probs[:, 1, :] > 0.5).long()
        else:  # non-deterministic
            feature_presence = torch.bernoulli(last_token_probs[:, 1, :]).long()

        batch_size, num_predicted = feature_presence.shape
        embedding_input = feature_presence.new_zeros(
            (batch_size, self.phonological_vocabulary_size)
        )
        embedding_input[:, :num_predicted] = feature_presence
        # A vector with every feature OFF decodes to [PAD]
        all_off = feature_presence.sum(1) == 0
        embedding_input[all_off, self.model_config.vocab.phon_pad_id] = 1

        return feature_presence, embedding_input

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

        The orthographic twin of :meth:`phonology_decoder_loop`: the batch stays dense
        inside the loop and the ragged ``orth_probs`` lists are cut once at the end.

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

        # Every sequence opens with a placeholder distribution that is certain of BOS.
        initial_prob = torch.zeros(
            (batch_size, self.orthographic_vocabulary_size),
            device=self.device,
        )
        initial_prob[:, bos_id] = 1

        step_probs: list[torch.Tensor] = []
        # A sequence stops collecting probabilities once it emits EOS, so the per-item
        # histories really are ragged. `sequence_finished` only ever flips on, so the rows
        # an item keeps are a prefix and this count is enough to split them back out.
        kept = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        sequence_finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.max_orth_seq_len - 1):
            # `sequence_finished` already holds what rescanning the whole generated history
            # for EOS would find, at one reduction instead of O(batch x length) per step.
            if bool(sequence_finished.all()):
                break

            step_mask = mask[: step + 1, : step + 1]

            orth_output = self.orthography_decoder(
                generated_orth_embeddings,
                memory=prompt_encoding,
                tgt_mask=step_mask,
            )

            # Only the last position is sampled, so project only that one: over a run the
            # full-prefix form does O(L^2) head work for O(L) usable rows.
            last_token_logits = self.linear_orthography_decoder(orth_output[:, -1])
            last_token_probs = torch.softmax(last_token_logits, dim=1)

            step_probs.append(last_token_probs)
            kept += (~sequence_finished).long()

            new_orthography_tokens = self.ortho_sample(last_token_probs, deterministic)

            # Append the one position just generated, at the slot it occupies: the
            # orthographic twin of the phonological loop's incremental embedding.
            generated_orth_embeddings = torch.cat(
                (
                    generated_orth_embeddings,
                    self.embed_orth_tokens(
                        new_orthography_tokens, position_offset=generated_orth_tokens.shape[1]
                    ),
                ),
                dim=1,
            )
            generated_orth_tokens = torch.cat(
                (generated_orth_tokens, new_orthography_tokens), dim=-1
            )

            sequence_finished = sequence_finished | (new_orthography_tokens == eos_id).squeeze(-1)

        if not step_probs:  # empty batch: every sequence is vacuously finished at step 0
            return [], generated_orth_tokens

        # Split the batched history back into ragged per-item lists, with a single device
        # sync for the whole call rather than one per item per step.
        probs = torch.stack(step_probs, dim=1)
        orth_probs = [[initial_prob[b], *probs[b, :n]] for b, n in enumerate(kept.tolist())]

        return orth_probs, generated_orth_tokens

    @torch.no_grad()
    def phonology_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_phon_embeddings: torch.Tensor,
        generated_phon_multihot: torch.Tensor,
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]], list[list[torch.Tensor]]]:
        """Autoregressive generation of phonological features.

        The batch stays dense inside the loop; the ragged ``list[list[Tensor]]`` shapes
        :class:`GenerationOutput` requires are cut once at the end. Each step embeds only
        the position it just generated, so the loop is linear in sequence length.

        Args:
            mask: Causal mask for decoder attention
            generated_phon_embeddings: (batch, 1, d_model) embedding of the seed prefix.
                Never recomputed; each step appends one row to it.
            generated_phon_multihot: (batch, 1, phon_vocab_size) seed feature vectors
                ([BOS]). Read once, after the loop, to prefix the generated history.
            prompt_encoding: Encoder context, (batch_size, d_embedding, d_model)
            deterministic: Sampling strategy flag
        """
        batch_size = prompt_encoding.size(0)
        phon_eos_id = self.model_config.vocab.phon_eos_id

        step_probs: list[torch.Tensor] = []
        step_vecs: list[torch.Tensor] = []
        step_multihot: list[torch.Tensor] = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for step in range(self.max_phon_seq_len - 1):
            step_mask = mask[: step + 1, : step + 1]
            phon_output = self.phonology_decoder(
                generated_phon_embeddings,
                memory=prompt_encoding,
                tgt_mask=step_mask,
            )

            batch, seq_len, _ = phon_output.shape
            # Only the last position is sampled; the width is spelled out rather than
            # inferred with -1, which is ambiguous for an empty batch.
            last_token_logits = self.linear_phonology_decoder(phon_output[:, -1]).view(
                batch, 2, self.phonological_vocabulary_size - 1
            )
            last_token_probs = torch.softmax(last_token_logits, dim=1)

            new_vectors, embedding_input = self.phono_sample(last_token_probs, deterministic)

            step_probs.append(last_token_probs[:, 1])  # Probability of feature being ON
            step_vecs.append(new_vectors)
            step_multihot.append(embedding_input)

            # Check for early stopping (if all sequences have hit EOS). One reduction per
            # step against a running flag, rather than rescanning the whole history.
            finished |= embedding_input[:, phon_eos_id].bool()
            if bool(finished.all()):
                break

            # Embed only the position just generated and append it. `seq_len` is the length
            # of the prefix just decoded, so it is also the index of the row being appended:
            # the one position embedding still owed. Placed after the break so the final,
            # never-read embedding is skipped.
            generated_phon_embeddings = torch.cat(
                (
                    generated_phon_embeddings,
                    self.embed_phon_vectors(embedding_input.unsqueeze(1), position_offset=seq_len),
                ),
                dim=1,
            )

        # Repack into the ragged per-batch-item shapes GenerationOutput expects. Every item
        # ran for the same number of steps, so this is a pure transpose.
        phon_probs = [list(item) for item in torch.stack(step_probs, dim=1)]
        phon_vecs = [list(item) for item in torch.stack(step_vecs, dim=1)]
        generated_phon_multihot = torch.cat(
            (generated_phon_multihot, torch.stack(step_multihot, dim=1)), dim=1
        )
        # One nonzero over the flattened buffer, then split by per-position feature count.
        # Calling nonzero per position instead costs a device sync apiece, which at
        # batch x length is more round trips than the whole loop above makes.
        flat = generated_phon_multihot.flatten(0, 1)
        active = torch.split(flat.nonzero()[:, 1], flat.sum(1).tolist())
        seq_len = generated_phon_multihot.shape[1]
        phon_tokens = [
            list(active[start : start + seq_len]) for start in range(0, len(active), seq_len)
        ]

        return phon_probs, phon_vecs, phon_tokens

    def _generate(
        self,
        pathway: Pathway,
        orth_enc_input: torch.Tensor | None = None,
        orth_enc_pad_mask: torch.Tensor | None = None,
        phon_enc_input: torch.Tensor | None = None,
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
            A ``GenerationDict``. ``global_encoding``, the (batch_size, d_embedding,
            d_model) memory passed to the decoders, is always populated; the
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
            orth_enc_input.size(0) if orth_enc_input is not None else phon_enc_input.size(0)  # type: ignore[union-attr]
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

                # Generation works in feature space, not phoneme-row space: the decoder
                # samples features independently, so a generated vector need not be any
                # real phoneme.
                generated_phon_multihot = torch.zeros(
                    (batch_size, 1, self.phonological_vocabulary_size),
                    dtype=torch.long,
                    device=self.device,
                )
                generated_phon_multihot[:, 0, self.model_config.vocab.phon_bos_id] = 1

                generated_phon_embeddings = self.embed_phon_vectors(generated_phon_multihot)
                phon_probs, phon_vecs, phon_tokens = self.phonology_decoder_loop(
                    mask,
                    generated_phon_embeddings,
                    generated_phon_multihot,
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

    def _validate_encoder_inputs(
        self,
        name: str,
        ids: torch.Tensor | None,
        mask: torch.Tensor | None,
        *,
        max_seq_len: int | None = None,
        tensor_exc: type[Exception] = ValueError,
    ) -> None:
        """Structural checks for one modality's encoder input and its padding mask.

        Both modalities carry ``(batch, sequence)`` integer ids and a boolean mask of the
        same shape, so one check serves them; ``name`` only selects the message prefix.

        ``tensor_exc`` selects the exception for an argument that is not a tensor at all.
        It is ``ValueError`` everywhere except ``op2op``'s orthographic inputs, which
        raise ``TypeError``. That one asymmetry predates this validator and is pinned by
        ``tests/domain/model/test_validate_generate_input.py``. ``max_seq_len`` enables the
        length bound, which only the pathways that own the sequence check.
        """
        if not isinstance(ids, torch.Tensor):
            raise tensor_exc(f"{name}_enc_input must be a torch.Tensor, got {type(ids)}")
        if ids.dim() != 2:
            raise ValueError(
                f"Expected 2D input tensor for {name}_enc_input, got shape: {tuple(ids.shape)}"
            )
        if ids.dtype not in (torch.long, torch.int):
            raise ValueError(
                f"{name}_enc_input must have dtype torch.long or torch.int, got {ids.dtype}"
            )
        if max_seq_len is not None and ids.size(1) > max_seq_len:
            raise ValueError(
                f"{name}_enc_input sequence length {ids.size(1)} exceeds "
                f"maximum allowed length {max_seq_len}"
            )

        if not isinstance(mask, torch.Tensor):
            raise tensor_exc(f"{name}_enc_pad_mask must be a torch.Tensor, got {type(mask)}")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name}_enc_pad_mask must have dtype torch.bool, got {mask.dtype}")
        if ids.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: {name}_enc_input is {tuple(ids.shape)} but "
                f"{name}_enc_pad_mask is {tuple(mask.shape)}"
            )

    def _validate_phon_bounds(self, phon_enc_input: torch.Tensor) -> None:
        """Check phoneme row ids index the phoneme table.

        Row space, not feature space: the bound is the number of phonemes (~91), not the
        phonological vocabulary size (~36).
        """
        num_rows = self.phon_feature_matrix.shape[0]
        if torch.any(phon_enc_input >= num_rows) or torch.any(phon_enc_input < 0):
            raise ValueError(f"Phoneme row ids must lie in [0, {num_rows})")

    def _validate_orth_bounds(self, orth_enc_input: torch.Tensor, label: str) -> None:
        """Check orthographic token ids fit the orthographic vocabulary."""
        if torch.any(orth_enc_input < 0):
            raise ValueError(f"{label} cannot be negative")
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
        phon_enc_input: torch.Tensor | None,
        phon_enc_pad_mask: torch.Tensor | None,
    ) -> None:
        """Validate the encoder inputs against the selected pathway.

        Each branch below states only what is specific to its pathway: which
        modalities are required, which must be absent, and which bounds/device
        checks apply. The per-modality structural checks are shared via
        :meth:`_validate_encoder_inputs`.
        """
        if pathway not in PATHWAYS:
            raise ValueError(f"Invalid pathway: {pathway}")

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
            self._validate_encoder_inputs(
                "phon", phon_enc_input, phon_enc_pad_mask, max_seq_len=self.max_phon_seq_len
            )
            self._validate_device(phon_enc_pad_mask=phon_enc_pad_mask)
            self._validate_phon_bounds(phon_enc_input)

        elif pathway == "o2p":
            if orth_enc_input is None:
                raise ValueError("orth_enc_input is required for o2p pathway")
            if orth_enc_pad_mask is None:
                raise ValueError("orth_enc_pad_mask is required for o2p pathway")
            self._validate_encoder_inputs("orth", orth_enc_input, orth_enc_pad_mask)

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
            self._validate_encoder_inputs("orth", orth_enc_input, orth_enc_pad_mask)
            self._validate_orth_bounds(orth_enc_input, "Input tokens")
            self._validate_device(
                orth_enc_input=orth_enc_input, orth_enc_pad_mask=orth_enc_pad_mask
            )

        else:  # op2op, the only pathway consuming both modalities
            if orth_enc_input is None or orth_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires orthographic inputs (orth_enc_input, orth_enc_pad_mask)"
                )
            if phon_enc_input is None or phon_enc_pad_mask is None:
                raise ValueError(
                    "op2op pathway requires phonological inputs (phon_enc_input, phon_enc_pad_mask)"
                )
            self._validate_encoder_inputs(
                "orth",
                orth_enc_input,
                orth_enc_pad_mask,
                max_seq_len=self.max_orth_seq_len,
                tensor_exc=TypeError,
            )
            self._validate_encoder_inputs(
                "phon", phon_enc_input, phon_enc_pad_mask, max_seq_len=self.max_phon_seq_len
            )
            if orth_enc_input.size(0) != phon_enc_input.size(0):
                raise ValueError(
                    f"Batch size mismatch: orthographic input has {orth_enc_input.size(0)} items "
                    f"but phonological input has {phon_enc_input.size(0)} items"
                )
            self._validate_phon_bounds(phon_enc_input)
            self._validate_orth_bounds(orth_enc_input, "Orthographic tokens")
