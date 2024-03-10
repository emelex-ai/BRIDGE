import torch


class Encoder(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(Encoder, self).__init__()
        # Set FF layer to 4*d_model
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4 * d_model
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Pass the input throught the various layers

        Args:
        src (Tensor) – the sequence to the encoder (required).
        src_mask (Optional[Tensor]) – the mask for the src sequence (optional). Defaults to None.
        src_key_padding_mask (Optional[Tensor]) – the mask for the src keys per batch (optional). Defaults to None.

        Returns:
            tensor
        """
        output = self.transformer_encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask
        )
        return output


class Decoder(torch.nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4 * d_model
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        """
        Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt (Tensor) – the sequence to the decoder (required).
            memory (Tensor) – the sequence from the last layer of the encoder (required).
            tgt_mask (Optional[Tensor]) – the mask for the tgt sequence (optional).
            memory_mask (Optional[Tensor]) – the mask for the memory sequence (optional).
            tgt_key_padding_mask (Optional[Tensor]) – the mask for the tgt keys per batch (optional).
            memory_key_padding_mask (Optional[Tensor]) – the mask for the memory keys per batch

        Returns:
            tensor

        """
        # print("tgt: ", tgt.shape)
        # print("memory: ", memory.shape)
        # print("tgt_mask: ", tgt_mask.shape)
        # print("tgt_mask: ", tgt_mask)
        # print("memory_mask: ", memory_mask.shape)
        # print("tgt_key_padding_mask: ", tgt_key_padding_mask.shape)
        # print("memory_key_padding_mask: ", memory_key_padding_mask.shape)
        output = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return output


# ----------------------------------------------------------------------
class Model(torch.nn.Module):
    def __init__(self, config, dataset):

        super().__init__()

        orth_vocab_size = len(dataset.character_tokenizer)
        phon_vocab_size = len(dataset.phonology_tokenizer)
        self.d_model = config.d_model
        self.d_embedding = config.d_embedding
        self.max_orth_seq_len = dataset.max_orth_seq_len
        self.max_phon_seq_len = dataset.max_phon_seq_len
        nhead = config.nhead
        nlayers_phon_enc = config.num_phon_enc_layers
        nlayers_orth_enc = config.num_orth_enc_layers
        nlayers_mixing_enc = config.num_mixing_enc_layers
        nlayers_phon_dec = config.num_phon_dec_layers
        nlayers_orth_dec = config.num_orth_dec_layers

        # Initial embeddings for orthography, phonology, and position
        # Embedding for orthography
        self.orthography_embedding = torch.nn.Embedding(orth_vocab_size, self.d_model)
        self.orth_position_embedding = torch.nn.Embedding(
            self.max_orth_seq_len, self.d_model
        )
        # Embedding for phonology
        self.phonology_embedding = torch.nn.Embedding(phon_vocab_size, self.d_model)
        self.phon_position_embedding = torch.nn.Embedding(
            self.max_phon_seq_len, self.d_model
        )
        self.vocab_sizes = {
            "orth_vocab_size": orth_vocab_size,
            "phon_vocab_size": phon_vocab_size,
        }

        # A 1 × d_embedding × d_model tensor of model parameters, rescaled by √d_model
        self.global_embedding = torch.nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model)) / self.d_model**0.5,
            requires_grad=True,
        )

        # Initial, encoding segment of our ConnTextUL model:
        # Instance of our Encoder module (defined above), for encoding orthography
        self.orthography_encoder = Encoder(
            d_model=self.d_model, nhead=nhead, num_layers=nlayers_orth_enc
        )
        # Instance of our Encoder module (defined above), for encoding phonology
        self.phonology_encoder = Encoder(
            d_model=self.d_model, nhead=nhead, num_layers=nlayers_phon_enc
        )

        # Criss-crossing orthography/phonology cross-attenion segment of ConnTextUL model
        self.gp_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=nhead, batch_first=True
        )
        self.gp_layer_norm = torch.nn.LayerNorm(self.d_model)
        self.pg_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=nhead, batch_first=True
        )
        self.pg_layer_norm = torch.nn.LayerNorm(self.d_model)

        # Segment of ConnTextUL model that mixes orthography/phonology representation
        self.transformer_mixer = Encoder(
            d_model=self.d_model,
            nhead=nhead,
            num_layers=nlayers_mixing_enc,
        )
        self.reduce = torch.nn.Linear(self.d_model, self.d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(self.d_model)

        # Decoder segment of ConnTextUL model
        # Orthography component of Decoder segment
        self.orthography_decoder = Decoder(
            d_model=self.d_model, nhead=nhead, num_layers=nlayers_orth_dec
        )
        self.linear_orthography_decoder = torch.nn.Linear(
            self.d_model, self.vocab_sizes["orth_vocab_size"]
        )
        # Phonology component of Decoder segment
        self.phonology_decoder = Decoder(
            d_model=self.d_model, nhead=nhead, num_layers=nlayers_phon_dec
        )
        # GE 2023-05-26:  Why the factor 2? Why the name linear?
        self.linear_phonology_decoder = torch.nn.Linear(
            self.d_model, 2 * (self.vocab_sizes["phon_vocab_size"] - 1)
        )

    # Returns a size×size, strictly upper-triangular Boolean tensor
    def generate_triangular_mask(self, size, device):
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), 1)
        return mask

    # -----------------------------------------------------
    def embed_orth_tokens(self, tokens):
        assert isinstance(
            tokens, torch.Tensor
        ), "For orthographic embeddings, tokens must be a pytorch tensor of integers (indices of orthography_embedding)"
        assert (
            tokens.dtype == torch.long or tokens.dtype == torch.int
        ), f"Input tensor to Embedding must be type int or long but is {tokens.dtype}"
        return (
            self.orthography_embedding(tokens)
            + self.orth_position_embedding.weight[None, : tokens.shape[1]]
        )

    # -----------------------------------------------------
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

        # Why do this? Isn't the device known in the beginning of the code?
        device = next(self.parameters()).device  # device of weights
        # len(tokens) is the batch size. self.d_model=16
        output_embedding = torch.zeros(
            (len(tokens), len(tokens[0]), self.d_model), device=device
        )
        for batch_num, batch in enumerate(tokens):
            for indx, tokes in enumerate(batch):
                # Here tokens should be a pytorch tensor of integers.
                # It extracts the indicated rows from self.phonology_embedding
                # ERROR tokes is empty. SOMETHING WRONG.
                avg_embedding = self.phonology_embedding(tokes).mean(axis=0)
                # Insert the resulting averaged embedding vector into the
                # output_embedding tensor as a new row
                output_embedding[batch_num, indx, :] = avg_embedding
        # Why is phon_position_embedding shape[1] not increasing? (BECAUSE thERE IS NO OUTPUT TOKEN)
        return (
            output_embedding  # (1,9,16), (1,9,16)
            # ERROR, (1,9,16), (1,10,16)
            + self.phon_position_embedding.weight[None, : len(tokens[0])]
        )

    # ----------------------------------------------------------------------

    def embed_o2p(self, orthography, orthography_padding_mask):
        orthography = self.embed_orth_tokens(orthography)
        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orthography_padding_mask
        )
        global_embedding = self.global_embedding.repeat(
            orthography_encoding.shape[0], 1, 1
        )
        orthography_encoding = torch.cat(
            (global_embedding, orthography_encoding), dim=1
        )
        orthography_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (orthography_encoding.shape[0], self.d_embedding),
                    device=orthography_encoding.device,
                    dtype=torch.bool,
                ),
                orthography_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(
            orthography_encoding, src_key_padding_mask=orthography_encoding_padding_mask
        )

        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

    def embed_p2o(self, phonology, phonology_padding_mask):
        phonology = self.embed_phon_tokens(phonology)
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phonology_padding_mask
        )
        global_embedding = self.global_embedding.repeat(
            phonology_encoding.shape[0], 1, 1
        )
        phonology_encoding = torch.cat((global_embedding, phonology_encoding), dim=1)
        phonology_encoding_padding_mask = torch.cat(
            (
                torch.zeros(
                    (phonology_encoding.shape[0], self.d_embedding),
                    device=phonology_encoding.device,
                    dtype=torch.bool,
                ),
                phonology_padding_mask,
            ),
            dim=-1,
        )

        mixed_encoding = self.transformer_mixer(
            phonology_encoding, src_key_padding_mask=phonology_encoding_padding_mask
        )

        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

    def embed_op2op(
        self, orthography, orthography_padding_mask, phonology, phonology_padding_mask
    ):
        orthography = self.embed_orth_tokens(orthography)
        phonology = self.embed_phon_tokens(phonology)

        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orthography_padding_mask
        )
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phonology_padding_mask
        )
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
        gp_pg_padding_mask = torch.cat(
            (orthography_padding_mask, phonology_padding_mask), dim=-1
        )

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

        mixed_encoding = self.transformer_mixer(
            gp_pg, src_key_padding_mask=gp_pg_padding_mask
        )

        # final_encoding = (self.reduce(mixed_encoding[:, :self.d_embedding]).unsqueeze(-2) + global_embedding)
        # final_encoding = self.reduce_layer_norm(final_encoding)

        # Add a residual connection to the final encoding
        final_encoding = mixed_encoding[:, : self.d_embedding] + global_embedding

        return final_encoding

    def forward(
        self,
        pathway,
        orth_enc_input,
        orth_enc_pad_mask,
        orth_dec_input,
        orth_dec_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    ):
        if pathway == "op2op":
            mixed_encoding = self.embed_op2op(
                orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
            )
            orth_dec_input = self.embed_orth_tokens(orth_dec_input)
            orth_ar_mask = self.generate_triangular_mask(
                orth_dec_input.shape[1], orth_dec_input.device
            )
            orth_output = self.orthography_decoder(
                tgt=orth_dec_input,
                tgt_mask=orth_ar_mask,
                tgt_key_padding_mask=orth_dec_pad_mask,
                memory=mixed_encoding,
            )
            phon_dec_input = self.embed_phon_tokens(phon_dec_input)
            phon_ar_mask = self.generate_triangular_mask(
                phon_dec_input.shape[1], phon_dec_input.device
            )
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
        elif pathway == "o2p":
            mixed_encoding = self.embed_o2p(orth_enc_input, orth_enc_pad_mask)
            phon_dec_input = self.embed_phon_tokens(phon_dec_input)  # , "p")
            phon_ar_mask = self.generate_triangular_mask(
                phon_dec_input.shape[1], phon_dec_input.device
            )
            phon_output = self.phonology_decoder(
                tgt=phon_dec_input,
                tgt_mask=phon_ar_mask,
                tgt_key_padding_mask=phon_dec_pad_mask,
                memory=mixed_encoding,
            )
            B, PC, E = phon_output.shape
            phon_token_logits = self.linear_phonology_decoder(phon_output)
            phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
            return {"phon": phon_token_logits}
        elif pathway == "p2o":
            mixed_encoding = self.embed_p2o(phon_enc_input, phon_enc_pad_mask)
            orth_dec_input = self.embed_orth_tokens(orth_dec_input)  # , "o")
            orth_ar_mask = self.generate_triangular_mask(
                orth_dec_input.shape[1], orth_dec_input.device
            )
            orth_output = self.orthography_decoder(
                tgt=orth_dec_input,
                tgt_mask=orth_ar_mask,
                tgt_key_padding_mask=orth_dec_pad_mask,
                memory=mixed_encoding,
            )
            orth_token_logits = self.linear_orthography_decoder(orth_output)
            orth_token_logits = orth_token_logits.transpose(1, 2)
            return {"orth": orth_token_logits}

    def size(self):
        param_num = 0
        param_size = 0
        for param in self.parameters():
            param_num += param.nelement()
            param_size += param.nelement() * param.element_size()
        buffer_num = 0
        buffer_size = 0
        for buffer in self.buffers():
            buffer_num += buffer.nelement()
            buffer_size += buffer.nelement() * buffer.element_size()

        param_num_all = param_num + buffer_num
        size_all_mb = round((param_size + buffer_size) / 1024**2, 3)
        result = {"parameters": param_num_all, "size in MB": size_all_mb}
        return result

    def ortho_sample(self, last_token_probs, deterministic):
        if deterministic:
            # Max probability token across the batch
            tokens = last_token_probs.argmax(dim=-1)
        else:
            # Random sampling based on the probability distributions for each item in the batch
            tokens = torch.multinomial(
                torch.softmax(last_token_probs, dim=-1), num_samples=1
            ).squeeze(-1)
        return tokens

    def phono_sample(self, last_token_probs, deterministic):
        """
        Sample from phonological decoder output. last_token_probs is a tensor of shape (max_seq_length, 2)
        where max_seq_length is the maximum sequence length of the phonological decoder and the 2 represents the
        probability dimension. For the probabilitye dimension (2) the zeroth index is the probability of the
        feature being off, and the first index is the probability of the feature being on.

        For example [0.6, 0.4] -> [feature off, feature on] and in this scenario the feature is off
        """

        if deterministic:
            vec = last_token_probs[:, 1] > 0.5
        else:  # non-deterministic
            vec = torch.bernoulli(last_token_probs[:, 1])

        tokens = torch.where(vec)[1]
        # If all features are off, then just return the BOS token
        if len(tokens) == 0:
            vec = torch.zeros_like(vec)
            vec[0, 31] = 1
            tokens = torch.tensor([31])

        return vec, tokens

    def phono_sample(self, last_token_probs, deterministic):
        batch_size, _, num_features = last_token_probs.shape
        if deterministic:
            # Greater than 0.5 probability indicates feature presence
            feature_presence = (last_token_probs[:, :, 1] > 0.5).long()
        else:
            # Sample each phonological feature based on its Bernoulli distribution
            feature_presence = torch.bernoulli(last_token_probs[:, :, 1]).long()

        # Initialize vectors to represent phonological features; each feature can be either 0 or 1
        phon_vectors = torch.zeros(
            batch_size,
            num_features,
            dtype=torch.long,
            device=last_token_probs.device,
        )
        for i in range(batch_size):
            for j in range(num_features):
                phon_vectors[i, j] = feature_presence[i, j]

        return phon_vectors

        return output

    def orthography_decoder_loop(
        self,
        mask,
        generated_orth_embeddings,
        generated_orth_tokens,
        prompt_encoding,
        deterministic,
    ):
        max_length = generated_orth_embeddings.shape[1]
        batch_size = generated_orth_embeddings.shape[0]
        device = generated_orth_embeddings.device

        generated_orth_probs = torch.zeros(
            (batch_size, max_length, self.vocab_sizes["orth_vocab_size"]),
            device=device,
        )

        for step in range(1, max_length):  # Start from 1 because 0 is already filled
            step_mask = mask[: step + 1, : step + 1].expand(batch_size, -1, -1)

            with torch.no_grad():
                orth_output = self.orthography_decoder(
                    tgt=generated_orth_embeddings[
                        :, :step
                    ],  # Process up to current step
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                    tgt_key_padding_mask=None,  # Assuming no padding within the generated sequence for simplicity
                )

                orth_token_logits = self.linear_orthography_decoder(orth_output)
                orth_token_probs = torch.softmax(orth_token_logits, dim=-1)

                last_token_probs = orth_token_probs[
                    :, -1, :
                ]  # Get the last timestep's probabilities
                generated_orth_probs[:, step, :] = last_token_probs

                if deterministic:
                    new_token = last_token_probs.argmax(dim=-1, keepdim=True)
                else:
                    new_token = torch.multinomial(last_token_probs, num_samples=1)

                generated_orth_tokens = torch.cat(
                    (generated_orth_tokens, new_token), dim=-1
                )
                generated_orth_embeddings = self.embed_orth_tokens(
                    generated_orth_tokens
                )

                # Update embeddings with new token for next step; not needed if we're on the last iteration
                if step < max_length - 1:
                    generated_orth_embeddings = torch.cat(
                        (
                            generated_orth_embeddings,
                            self.embed_orth_tokens(new_token[:, None]).expand(
                                -1, max_length, -1
                            ),
                        ),
                        dim=1,
                    )

        # After loop ends, we should have a batch of generated tokens and their corresponding probabilities
        return {
            "orth_probs": generated_orth_probs,  # [Batch, Seq Length, Vocab Size]
            "orth_tokens": generated_orth_tokens,  # [Batch, Seq Length]
        }

        return output

    def phonology_decoder_loop(
        self,
        mask,
        generated_phon_embeddings,
        generated_phon_tokens,
        prompt_encoding,
        deterministic,
    ):
        print("--In phonology_decoder_loop--")
        print(f"generated_phon_embeddings: {generated_phon_embeddings.shape}")
        print(f"generated_phon_tokens: {generated_phon_tokens}")
        print(f"prompt_encoding: {prompt_encoding.shape}")
        max_length = generated_phon_embeddings.shape[1]
        batch_size = generated_phon_embeddings.shape[0]
        device = generated_phon_embeddings.device

        generated_phon_probs = torch.zeros(
            (
                batch_size,
                max_length,
                2,
                self.phonology_embedding.num_embeddings - 1,
            ),
            device=device,
        )  # Assuming phonology vocab size for -1 adjustment
        generated_phon_vectors = []

        for step in range(
            1, max_length
        ):  # Start from 1 to account for initial BOS token
            step_mask = self.generate_triangular_mask(step + 1, device).expand(
                batch_size, -1, -1
            )

            with torch.no_grad():
                phon_output = self.phonology_decoder(
                    tgt=generated_phon_embeddings[
                        :, :step
                    ],  # Process up to current step
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                    tgt_key_padding_mask=None,  # Assuming no padding within the generated sequence for simplicity
                )
                phon_token_logits = self.linear_phonology_decoder(phon_output)
                phon_token_logits = phon_token_logits.view(batch_size, step, 2, -1)
                phon_token_probs = torch.softmax(phon_token_logits, dim=-1)

                last_token_probs = phon_token_probs[
                    :, -1, :
                ]  # Get the last timestep's probabilities
                generated_phon_probs[:, step, :, :] = last_token_probs

                if deterministic:
                    new_tokens = last_token_probs.argmax(dim=-1)
                else:
                    # Sampling based on probabilities for on/off states for each feature
                    new_tokens = torch.bernoulli(last_token_probs[:, :, 1]).long()

                generated_phon_tokens.append(
                    new_tokens
                )  # This needs to be appended to some batched structure
                generated_phon_vectors.append(
                    new_tokens
                )  # This might need rethinking on how to effectively batch

                # Update embeddings with new tokens for next step; not needed if we're on the last iteration
                if step < max_length - 1:
                    # This operation might need adjustment based on the correct shape of new_tokens
                    generated_phon_embeddings = torch.cat(
                        (
                            generated_phon_embeddings,
                            self.embed_phon_tokens([new_tokens]).expand(
                                -1, max_length, -1
                            ),
                        ),
                        dim=1,
                    )

        # After loop ends, we should have a batch of generated tokens and their corresponding probabilities
        # Convert lists to tensors for consistency with batch processing
        print(f"generated_phon_tokens: {generated_phon_tokens}")
        generated_phon_tokens_tensor = torch.stack(generated_phon_tokens, dim=1)
        generated_phon_vectors_tensor = torch.stack(generated_phon_vectors, dim=1)

        return {
            "phon_probs": generated_phon_probs,  # [Batch, Seq Length, 2, Phonology Vocab Size]
            "phon_tokens": generated_phon_tokens_tensor,  # [Batch, Seq Length]
            "phon_vecs": generated_phon_vectors_tensor,  # [Batch, Seq Length, Phonology Feature Size]
        }

    def generate(
        self,
        pathway,
        orth_enc_input,
        orth_enc_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        deterministic=False,
    ):
        self.eval()
        device = next(self.parameters()).device
        batch_size = orth_enc_input.shape[0]
        print(f"--In generate--")
        print(f"orth_enc_input: {orth_enc_input.shape}")
        print(f"orth_enc_pad_mask: {orth_enc_pad_mask.shape}")
        print(f"phon_enc_input: {phon_enc_input.shape}")
        print(f"phon_enc_pad_mask: {phon_enc_pad_mask.shape}")

        with torch.no_grad():
            if pathway == "op2op":
                prompt_encoding = self.embed_op2op(
                    orth_enc_input,
                    orth_enc_pad_mask,
                    phon_enc_input,
                    phon_enc_pad_mask,
                )
            elif pathway == "o2p":
                prompt_encoding = self.embed_o2p(orth_enc_input, orth_enc_pad_mask)
            elif pathway == "p2o":
                prompt_encoding = self.embed_p2o(phon_enc_input, phon_enc_pad_mask)

        # Prepare for batch processing
        if pathway in ["op2op", "p2o"]:
            orth_dec_input = self.embed_orth_tokens(
                torch.full(
                    (batch_size, 1),
                    self.character_tokenizer.char_2_idx["[BOS]"],
                    dtype=torch.long,
                    device=device,
                )
            )
            orth_mask = self.generate_triangular_mask(self.max_orth_seq_len, device)
            orth_output = self.orthography_decoder_loop(
                orth_mask,
                orth_dec_input,
                None,  # Placeholder, as tokens are managed within the loop
                prompt_encoding,
                deterministic,
            )
        else:
            orth_output = {"orth_probs": None, "orth_tokens": None}

        if pathway in ["op2op", "o2p"]:
            phon_dec_input = torch.full(
                (batch_size, 1), 31, dtype=torch.long, device=device
            )  # BOS token for phonology
            print(f"{phon_dec_input.shape=}")
            print(f"{phon_dec_input=}")
            phon_mask = self.generate_triangular_mask(self.max_phon_seq_len, device)
            phon_output = self.phonology_decoder_loop(
                phon_mask,
                phon_dec_input,  # Initial BOS token embedded
                None,  # Placeholder, as tokens are managed within the loop
                prompt_encoding,
                deterministic,
            )
        else:
            phon_output = {
                "phon_probs": None,
                "phon_vecs": None,
                "phon_tokens": None,
            }

        return {
            "orth_probs": orth_output.get("orth_probs"),
            "orth_tokens": orth_output.get("orth_tokens"),
            "phon_probs": phon_output.get("phon_probs"),
            "phon_vecs": phon_output.get("phon_vecs"),
            "phon_tokens": phon_output.get("phon_tokens"),
            "global_encoding": prompt_encoding,
        }
