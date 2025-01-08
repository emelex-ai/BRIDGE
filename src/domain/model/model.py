from src.domain.datamodels import DatasetConfig, ModelConfig
from src.domain.model.encoder import Encoder
from src.domain.model.decoder import Decoder
<<<<<<< HEAD
from torch.nn.utils.rnn import pad_sequence
from src.utils.helper_funtions import set_seed
from typing import Dict, List, Union
from itertools import accumulate
=======
from src.utils.helper_funtions import set_seed
from typing import List, Union, Tuple
>>>>>>> main
import torch.nn as nn
import torch


class Model(nn.Module):
<<<<<<< HEAD
    def __init__(self, model_config: ModelConfig, dataset_config: DatasetConfig, device: torch.device = "cuda") -> None:
=======
    def __init__(
        self,
        model_config: ModelConfig,
        dataset_config: DatasetConfig,
        device: str,
    ) -> None:
>>>>>>> main
        super().__init__()
        self.device = torch.device(device)
        self.d_model: int = model_config.d_model
        self.d_embedding: int = model_config.d_embedding
        self.max_orth_seq_len: int = dataset_config.max_orth_seq_len
        self.max_phon_seq_len: int = dataset_config.max_phon_seq_len
        self.nhead: int = model_config.nhead

        if model_config.seed:
            set_seed(seed=model_config.seed)
        # Initialize embeddings and position embeddings
        self.orthography_embedding = nn.Embedding(
            dataset_config.orthographic_vocabulary_size, self.d_model
        )
        self.orth_position_embedding = nn.Embedding(self.max_orth_seq_len, self.d_model)
        self.phonology_embedding = nn.Embedding(
            dataset_config.phonological_vocabulary_size, self.d_model
        )
        self.phon_position_embedding = nn.Embedding(self.max_phon_seq_len, self.d_model)

        self.global_embedding = nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model), device=self.device)
            / self.d_model**0.5,
            requires_grad=True,
        )
        self.orthography_encoder = Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=model_config.num_orth_enc_layers,
        )
<<<<<<< HEAD
        self.orthography_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_orth_enc_layers
        )

        self.phonology_encoder = Encoder(
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_phon_enc_layers
=======

        self.phonology_encoder = Encoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=model_config.num_phon_enc_layers,
>>>>>>> main
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
<<<<<<< HEAD
            d_model=self.d_model, nhead=self.nhead, num_layers=model_config.num_mixing_enc_layers
=======
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=model_config.num_mixing_enc_layers,
>>>>>>> main
        )

        self.reduce = torch.nn.Linear(self.d_model, self.d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(self.d_model)

        # Decoders and output layers
        self.orthography_decoder = Decoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=model_config.num_orth_dec_layers,
        )
<<<<<<< HEAD
        self.linear_orthography_decoder = nn.Linear(self.d_model, dataset_config.orthographic_vocabulary_size)
=======
        self.linear_orthography_decoder = nn.Linear(
            self.d_model, dataset_config.orthographic_vocabulary_size
        )
>>>>>>> main

        self.phonology_decoder = Decoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=model_config.num_phon_dec_layers,
        )
        self.linear_phonology_decoder = nn.Linear(
            self.d_model, 2 * (dataset_config.phonological_vocabulary_size - 1)
        )
<<<<<<< HEAD
        self.linear_phonology_decoder = nn.Linear(self.d_model, 2 * (dataset_config.phonological_vocabulary_size - 1))

    # Helper functions
    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.orthography_embedding(tokens) + self.orth_position_embedding.weight[None, : tokens.shape[1]]
=======

    # Helper functions
    def embed_orth_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return (
            self.orthography_embedding(tokens)
            + self.orth_position_embedding.weight[None, : tokens.shape[1]]
        )
>>>>>>> main

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
<<<<<<< HEAD
        device = next(self.parameters()).device  # device of weights
        # len(tokens) is the batch size
        output_embedding = torch.zeros((batch_size, max_phon_len, self.d_model), device=device)
=======
        # len(tokens) is the batch size
        output_embedding = torch.zeros(
            (batch_size, max_phon_len, self.d_model), device=self.device
        )
>>>>>>> main
        for batch_num, batch in enumerate(tokens):
            for indx, tokes in enumerate(batch):
                # Here tokens should be a pytorch tensor of integers.
                # It extracts the indicated rows from self.phonology_embedding
                avg_embedding = self.phonology_embedding(tokes).mean(axis=0)
                # Insert the resulting averaged embedding vector into the
                # output_embedding tensor as a new row
                output_embedding[batch_num, indx, :] = avg_embedding
<<<<<<< HEAD
        return output_embedding + self.phon_position_embedding.weight[None, : len(tokens[0])]
=======
        return (
            output_embedding
            + self.phon_position_embedding.weight[None, : len(tokens[0])]
        )
>>>>>>> main

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
<<<<<<< HEAD
        orthography = self.embed_orth_tokens(orth_enc_input)  # Shape: (batch_size, seq_len, d_model)
        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orth_enc_pad_mask)
=======
        orthography = self.embed_orth_tokens(
            orth_enc_input
        )  # Shape: (batch_size, seq_len, d_model)
        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orth_enc_pad_mask
        )
>>>>>>> main
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
<<<<<<< HEAD
            (batch_size, 1), device=orthography_encoding.device, dtype=torch.bool
=======
            (batch_size, 1), device=self.device, dtype=torch.bool
>>>>>>> main
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
        phon_dec_input: List[torch.Tensor],
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

<<<<<<< HEAD
    def embed_p(self, phon_enc_input: List[torch.Tensor], phon_enc_pad_mask: torch.Tensor):
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phon_enc_pad_mask)
        global_embedding = self.global_embedding.repeat(phonology_encoding.shape[0], 1, 1)
=======
    def embed_p(
        self, phon_enc_input: List[torch.Tensor], phon_enc_pad_mask: torch.Tensor
    ):
        phonology = self.embed_phon_tokens(phon_enc_input)
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phon_enc_pad_mask
        )
        global_embedding = self.global_embedding.repeat(
            phonology_encoding.shape[0], 1, 1
        )
>>>>>>> main
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        phonology_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (phonology_encoding.shape[0], self.d_embedding),
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
        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding
        return final_encoding

    def forward_p2o(
        self,
        phon_enc_input: List[torch.Tensor],
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
        phon_enc_input: List[torch.Tensor],
        phon_enc_pad_mask: torch.Tensor,
        phon_dec_input: List[torch.Tensor],
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
                    (gp_pg.shape[0], self.d_embedding),
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
        # print(orth_token_logits)
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
    ) -> Tuple[torch.Tensor, List[List[torch.Tensor]]]:
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
                        for that sample. If all features are off, we default to [33].

        Sample from phonological decoder output. last_token_probs is a tensor of shape (batch_size, 2, 33)
        where 2 represents the probability dimension and 33 is the number of possible phonological vector
        features (including BOS, EOS, PAD). For the probabilitye dimension (2) the zeroth index is the probability
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
        tokens_tmp = torch.where(feature_presence)

        # We create this new data structure to store the tokens in the list of lists. Each index of the outer list
        # corresponds to a batch element, and the inner list contains the indices of the active features for that
        # batch element.
        out_tokens = [[] for _ in range(last_token_probs.shape[0])]
        for x, y in zip(*tokens_tmp):
            out_tokens[x.item()].append(y.item())

        # If all features are off, then just return the PAD token
        for i, tokens in enumerate(out_tokens):
            if len(tokens) == 0:
                out_tokens[i] = torch.tensor([33])

        return feature_presence, out_tokens

    def orthography_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_orth_embeddings: torch.Tensor,
        generated_orth_tokens: torch.Tensor,
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ):
        """
        Iteratively decodes/generates orthographic tokens.

        Args:
            mask: (max_orth_seq_len, max_orth_seq_len) Triangular causal mask for the decoder.
            generated_orth_embeddings: Current embeddings for the partially generated tokens.
            generated_orth_tokens: (1, current_seq_len) store of all tokens so far.
            prompt_encoding: (batch_size, ..., embed_dim) The memory/context from the encoder(s).
            deterministic: Whether to sample next token greedily or stochastically.

        Returns:
            A dict containing:
                - "orth_probs": List of probability distributions at each step
                - "orth_tokens": Final list/tensor of generated token IDs
        """
        # Start by storing an initial probability distribution "placeholder" for illustration
        # (the user code had a zero vector with index 31 set to 1).
        # You might simply want to store the real distribution from step 0, but we'll keep
        # this example as-is.
        vec = torch.zeros(33)
        vec[31] = 1
        orth_probs = [vec]
        for step in range(self.max_orth_seq_len - 1):
            step_mask = mask[: step + 1, : step + 1]

            with torch.no_grad():
                orth_output = self.orthography_decoder(
                    generated_orth_embeddings,
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                )
                B, OC, E = orth_output.shape
                linear_output = self.linear_orthography_decoder(orth_output)
                orthography_token_logits = linear_output.transpose(1, 2)

                last_token_logits = orthography_token_logits[:, :, -1]

                last_token_probs = torch.softmax(last_token_logits, dim=1)
                orth_probs.append(last_token_probs[0])

                new_orthography_token = self.ortho_sample(
                    last_token_probs, deterministic
                )

                generated_orth_tokens = torch.cat(
                    (generated_orth_tokens, new_orthography_token), dim=-1
                )
                generated_orth_embeddings = self.embed_orth_tokens(
                    generated_orth_tokens
                )
                # If we have generated the end token, stop.
                if new_orthography_token == 4:
                    break

        output = {"orth_probs": orth_probs, "orth_tokens": generated_orth_tokens[0]}

        return output

    def phonology_decoder_loop(
        self,
        mask: torch.Tensor,
        generated_phon_embeddings: torch.Tensor,
        generated_phon_tokens: List[List[torch.Tensor]],
        prompt_encoding: torch.Tensor,
        deterministic: bool,
    ):
        """
        Iteratively decodes/generates phonological features for each batch element.
        Each batch element is a list of tensors representing (possibly multiple) phonological tokens.

        Args:
            mask: (max_phon_seq_len, max_phon_seq_len) Triangular causal mask.
            generated_phon_embeddings: Current embeddings for the partially generated phon features.
            generated_phon_tokens: A list of lists of Tensors: shape (batch_size, current_length_of_sequence).
            prompt_encoding: (batch_size, ..., embed_dim) The memory/context from the encoder(s).
            deterministic: Whether to sample features greedily (>0.5) or stochastically (Bernoulli).

        Returns:
            A dict containing:
                - "phon_probs": Probability (Bernoulli parameter) at each step for each feature
                - "phon_vecs": The 0/1 vectors actually used
                - "phon_tokens": Indices of features that were ON for each token
        """
        phon_probs = []
        phon_vecs = []
        for _ in range(generated_phon_embeddings.shape[0]):
            phon_probs.append([])
            phon_vecs.append([])

        for step in range(self.max_phon_seq_len - 1):
            step_mask = mask[: step + 1, : step + 1]

            with torch.no_grad():
                phon_output = self.phonology_decoder(
                    generated_phon_embeddings,
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                )

                B, PC, E = phon_output.shape
                phonology_token_logits = self.linear_phonology_decoder(phon_output)

                phonology_token_logits = phonology_token_logits.view(
                    B, PC, 2, -1
                ).transpose(1, 2)

                last_token_logits = phonology_token_logits[:, :, -1, :]

                last_token_probs = torch.softmax(last_token_logits, dim=1)
                for i, probs in enumerate(last_token_probs):
                    # Recall index 0 is probability of feature being off, index 1 is probability of feature being on
                    phon_probs[i].append(probs[1])

                new_phonology_vectors, new_phonology_tokens = self.phono_sample(
                    last_token_probs, deterministic
                )
                for gen_phon_tokes, new_phon_tokes in zip(
                    generated_phon_tokens, new_phonology_tokens
                ):
                    gen_phon_tokes.append(torch.tensor(new_phon_tokes))

                for i, vec in enumerate(new_phonology_vectors):
                    phon_vecs[i].append(vec)

                # generated_phon_tokens[0].append(new_phonology_tokens)
                generated_phon_embeddings = self.embed_phon_tokens(
                    generated_phon_tokens
                )

        output = {
            "phon_probs": phon_probs,
            "phon_vecs": phon_vecs,
            "phon_tokens": generated_phon_tokens,
        }

        return output

    def generate(
        self,
        pathway: str,
        orth_enc_input: torch.Tensor,
        orth_enc_pad_mask: torch.Tensor,
        phon_enc_input: torch.Tensor,
        phon_enc_pad_mask: torch.Tensor,
        deterministic: bool = False,
    ):
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
            - pathway (string): one of three pathway options ['op2op', 'o2p' 'p2o'].
            - orth_enc_input (torch.tensor): Output from the tokenizer encode routine. Has shape (batch_size, max_sen_len)
            - orth_enc_pad_mask (torch.tensor): Also output from the tokenizer encode routine. Has shape (batch_size, max_seq_len). This is a boolena tensor
                of Trues and Falses, where the True indicates the presence of a PAD token (the integer 4). See Example section for example
            - phon_enc_input (): TODO
            - phon_enc_pad_mask (): TODO
            - deterministic (boolean): Flag contolling whether the token sampling process is deteministic (greedy) or stochastic, meaning we draw from a bernoulli
                distribution parameterized by model's output probabilities.

        Returns:
            The routine returna a dictionary with keys containing the generated data at various levels, along with the global embedding vector.
            More precisely,

                >>> output = model.generate(**params)

                - output["orth_probs"] (List(torch.tensor)): TODO
                - output["orth_tokens"] (List(torch.tensor)): TODO
                - output["phon_probs"] (List(List(torch.tensor))): This object contains the probabilities (bernoulli parameters) of each feature vector, for each
                    phonome, for each word in the batch. So the shape is (batch_size, max_phon_len, 33) or (words, phonemes, phonological features).
                - output["phon_vecs"] (List(List(torch.tensor))): The phonological feature vectors resulting from the generation process. If determinisitc=True then
                    each tensor is created from the output probabilites:
                        (last_token_probs[:, 1, :] > 0.5).long())
                    if deterministic=False, then the vector is sampled:
                        torch.bernoulli(last_token_probs[:, 1, :]).long()
                    The final object has shape (batch_size, max_phon_len, 33) or (words, phonemes, phonological features).
                - output["phon_tokens"] (List(List(torch.tensor))): Each feature vector is composed of zeros and ones. We keep track of all the indices where phonological
                    features are on, or set to one, with this object. The outer list contains the words, the inner list contains all the feature vectors indices, each
                    inner torch.tensor contains numbers between 0 - 33, where each indicates the presence of a one at the corresponding index in the phonologica feature
                    vector. In other words, the shape is (batch_size, max_phon_len, num_active_features). Each inner tensor can have a different size, as its length
                    ie equal to the number of nonzero entries in the phonological feature vector.
                - output["global_encoding"] (torch.tensor): This is the global embedding vector that is passed into the decoder as the Key/Value tensors in the cross attention
                    head. Its shape is (batch_size, 1, global_embedding_dim), or (word, the vector, num elements in the vector).

        Examples:
            Below we generate phonological data for the three input words "hello", "dog", and "a".

            >>> from src.model import Model
            >>> from addict import Dict as AttrDict
            >>> from pathlib import Path

            >>> config = type(
                            "config",
                            (object,),
                            {"dataset_filename": Path("data/data.csv")},
                        )
            >>> ds = ConnTextULDataset(config)
            >>> chkpt = pt.load("path_to_checkpoint_file.pth")
            >>> model = Model(AttrDict(chkpt["config"]), ds)
            >>> model.load_state_dict(chkpt["model_state_dict"])

            >>> datum = ds.character_tokenizer.encode(["hello", "dog", "a"])
            >>> datum['enc_input_ids']
                tensor([[ 0, 18, 15, 22, 22, 25,  1],
                        [ 0, 14, 25, 17,  1,  4,  4],
                        [ 0, 11,  1,  4,  4,  4,  4]])
            >>> datum['enc_pad_mask']
                tensor([[False, False, False, False, False, False, False],
                        [False, False, False, False, False,  True,  True],
                        [False, False, False,  True,  True,  True,  True]])

            >>> pred = model.generate(
                            "o2p",
                            datum["enc_input_ids"],
                            datum["enc_pad_mask"],
                            None,
                            None,
                            deterministic=True,
                        )
            >>> pred.keys()
                dict_keys(['orth_probs', 'orth_tokens', 'phon_probs', 'phon_vecs', 'phon_tokens', 'global_encoding'])

        Note:
            Only the o2p pathway is currently implemented to support batch processing. Need to add an issue to complete implementation of the
            p2o and op2op batch processing pathways.

        See Also:
            - phonology_decoder_loop
            - phono_sample
            - orthography_decoder_loop
            - ortho_sample
        """
        self.eval()
        batch_size = orth_enc_input.shape[0]

        with torch.no_grad():
            if pathway == "op2op":
                prompt_encoding = self.embed_op(
                    orth_enc_input,
                    orth_enc_pad_mask,
                    phon_enc_input,
                    phon_enc_pad_mask,
                )
            elif pathway == "o2p":
                prompt_encoding = self.embed_o(orth_enc_input, orth_enc_pad_mask)
            elif pathway == "p2o":
                prompt_encoding = self.embed_p(phon_enc_input, phon_enc_pad_mask)

        generated_phon_probs = None
        generated_phon_vecs = None
        generated_phon_tokens = None
        if pathway in ["op2op", "o2p"]:
            mask = self.generate_triangular_mask(self.max_phon_seq_len, self.device)

            # Here we create the container to store all the generated tokens for each input word (batch_size).
            # We initialize each list in the list of lists (batches of words) with the BOS token (31). We will
            # append new tensors to these lists as we generate the phonological vectors. For example if we have
            # a batch size of 2, meaning 2 words have been input, then after 2
            # tokens have been generated the generated_phon_tokens list might look like this:
            # [
            #   [tensor([31]), tensor([2, 13]), tensor([9, 10])]
            #   [tensor([31]), tensor([4, 5, 21]), tensor([22])]
            # ]
            generated_phon_tokens = [
                [torch.tensor([31], dtype=torch.long, device=self.device)]
                for _ in range(batch_size)
            ]

            generated_phon_embeddings = self.embed_phon_tokens(generated_phon_tokens)
            generated_phon_output = self.phonology_decoder_loop(
                mask,
                generated_phon_embeddings,
                generated_phon_tokens,
                prompt_encoding,
                deterministic,
            )
            generated_phon_probs = generated_phon_output["phon_probs"]
            generated_phon_vecs = generated_phon_output["phon_vecs"]
            generated_phon_tokens = generated_phon_output["phon_tokens"]

        generated_orth_probs = None
        generated_orth_tokens = None
        if pathway in ["op2op", "p2o"]:
            mask = self.generate_triangular_mask(self.max_orth_seq_len, self.device)
            generated_orth_tokens = torch.tensor(
                [[0]], dtype=torch.long, device=self.device
            )
            generated_orth_embeddings = self.embed_orth_tokens(generated_orth_tokens)
            generated_orth_output = self.orthography_decoder_loop(
                mask,
                generated_orth_embeddings,
                generated_orth_tokens,
                prompt_encoding,
                deterministic,
            )
            generated_orth_probs = generated_orth_output["orth_probs"]
            generated_orth_tokens = generated_orth_output["orth_tokens"]

        output = {
            "orth_probs": generated_orth_probs,
            "orth_tokens": generated_orth_tokens,
            "phon_probs": generated_phon_probs,
            "phon_vecs": generated_phon_vecs,
            "phon_tokens": generated_phon_tokens,
            "global_encoding": prompt_encoding,
        }

        return output
