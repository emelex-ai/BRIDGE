import torch


class Encoder(torch.nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1):
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
    def __init__(self, d_model=512, nhead=1, num_layers=1):
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
        #print("tgt: ", tgt.shape)
        #print("memory: ", memory.shape)
        #print("tgt_mask: ", tgt_mask.shape)
        #print("tgt_mask: ", tgt_mask)
        #print("memory_mask: ", memory_mask.shape)
        #print("tgt_key_padding_mask: ", tgt_key_padding_mask.shape)
        #print("memory_key_padding_mask: ", memory_key_padding_mask.shape)
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
    def __init__(
        self,
        orth_vocab_size,
        phon_vocab_size,
        d_model,
        d_embedding,
        max_orth_seq_len,
        max_phon_seq_len,
        nhead,
        num_layers_dict,
    ):
        super().__init__()
        print("max_orth_seq_len: ", max_orth_seq_len)
        print("max_phon_seq_len: ", max_phon_seq_len)
        # print("ortho: vocabulary size: ", orth_vocab_size)  # 49
        # print("phono: vocabulary size: ", phon_vocab_size)  # 34

        nlayers_phon_enc = num_layers_dict["phon_enc"]
        nlayers_phon_dec = num_layers_dict["phon_dec"]
        nlayers_orth_enc = num_layers_dict["orth_enc"]
        nlayers_orth_dec = num_layers_dict["orth_dec"]
        nlayers_mixing_enc = num_layers_dict["mixing_enc"]

        # Initial embeddings for orthography, phonology, and position
        # Embedding for orthography
        self.orthography_embedding = torch.nn.Embedding(
            orth_vocab_size, d_model)
        self.orth_position_embedding = torch.nn.Embedding(
            max_orth_seq_len, d_model
        )  # GE  added an independent pos embedding
        print("orth position embedding: max_orth_seq_len: ", max_orth_seq_len)
        # Embedding for phonology
        self.phonology_embedding = torch.nn.Embedding(phon_vocab_size, d_model)
        self.phon_position_embedding = torch.nn.Embedding(
            max_phon_seq_len, d_model)  # GE

        self.vocab_sizes = {
            "orth_vocab_size": orth_vocab_size,
            "phon_vocab_size": phon_vocab_size,
        }
        self.d_model = d_model
        self.d_embedding = d_embedding
        self.max_orth_seq_len = max_orth_seq_len
        self.max_phon_seq_len = max_phon_seq_len
        self.max_seq_len = max(max_orth_seq_len, max_phon_seq_len)  # GE

        # A 1×1×d_model tensor of model parameters, rescaled by √d_model
        self.global_embedding = torch.nn.Parameter(
            torch.randn((1, self.d_embedding, self.d_model)) /
            self.d_model**0.5,
            requires_grad=True,
        )

        # Initial, encoding segment of our ConnTextUL model:
        # Instance of our Encoder module (defined above), for encoding orthography
        self.orthography_encoder = Encoder(
            d_model=d_model, nhead=nhead, num_layers=nlayers_orth_enc
        )
        # Instance of our Encoder module (defined above), for encoding phonology
        self.phonology_encoder = Encoder(
            d_model=d_model, nhead=nhead, num_layers=nlayers_phon_enc
        )

        # Criss-crossing orthography/phonology cross-attenion segment of ConnTextUL model
        self.gp_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.gp_layer_norm = torch.nn.LayerNorm(d_model)
        self.pg_multihead_attention = torch.nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, batch_first=True
        )
        self.pg_layer_norm = torch.nn.LayerNorm(d_model)

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
        mask = torch.triu(torch.ones(
            (size, size), dtype=torch.bool, device=device), 1)
        return mask

    # Returns embeddings
    def embed_tokens(self, tokens, mode="o"):
        assert mode in ["o", "p"]

        if mode == "o":
            assert isinstance(
                tokens, torch.Tensor
            ), "For orthographic embeddings, tokens must be a pytorch tensor of integers (indices of orthography_embedding)"
            assert (
                tokens.dtype == torch.long or tokens.dtype == torch.int
            ), f"Input tensor to Embedding must be type int or long but is {tokens.dtype}"
            # self.orthography_embedding = torch.nn.Embedding(orth_vocab_size, d_model)
            # print("tokens shape: ", tokens.shape)  # 5, 5  (batch = 5, max word size: 5) | 8, 7 (batch_size, num_tokens)
            # 5,5,64  |  8,7,16
            #print("=shape self.orthography_embedding(tokens): ", self.orthography_embedding(tokens).shape)
            # 1, 2, 64 | 1, 7, 16   # So in scratch pad error is 2nd dimension of position
            # print("tokens.shape[1]: ", tokens.shape[1])  # 7 (nb words)
            #print("=shape self.orth_position_embedding.weight[None, :tokens.shape[1]]: ", self.orth_position_embedding.weight[None, :tokens.shape[1]].shape)
            return (   # ERROR L197
                # ERROR: The size of tensor a (5) must match the size of tensor b (2) at non-singleton dimension 1
                self.orthography_embedding(tokens)   # ERROR
                + self.orth_position_embedding.weight[None, : tokens.shape[1]]
            )  # GE
        # This is where we need to average the phonological embedding vectors
        else:
            try:
                isinstance(tokens, list)
            except:
                raise TypeError(
                    f"For phonological vectors, tokens must be a list where each element is a pytorch tensor of integers (indices), but is type: {type(tokens)}"
                )
            try:
                all(isinstance(token, torch.Tensor) for token in tokens)
            except:
                for token in tokens:
                    print(f"type(token) = {type(token)}")
                    print(f"token = {token}")
                raise TypeError(
                    "For phonological vectors, each element of the list must be a pytorch tensor of integers (indices)"
                )
            # Here we average the embeddings for each feature in a phonological vector
            # Each row of indices will become of batch once we extract rows from the embedding matrix
            # So the size of the resulting 'output_embedding' tensor should be (batch_size, max_phon_len, d_model)
            device = next(self.parameters()).device
            output_embedding = torch.zeros(
                (len(tokens), len(tokens[0]), self.d_model), device=device
            )
            for batch_num, batch in enumerate(tokens):
                for indx, tokes in enumerate(batch):
                    # Here tokens should be a pytorch tensor of integers. It extracts the indicated rows from self.phonology_embedding
                    avg_embedding = self.phonology_embedding(
                        tokes).mean(axis=0)
                    # Insert the resulting averaged embedding vector into the output_embedding tensor as a new row
                    output_embedding[batch_num, indx, :] = avg_embedding
            print("output_embedding: ", output_embedding.shape)
            print("tokens: ", tokens)
            print("len tokens[0]: ", len(tokens[0]))
            print("shape phon_position_embedding: ", self.phon_position_embedding.weight[None, :len(tokens[0])].shape)
            #print("tokens.shape: ", tokens.shape)
            return (
                output_embedding
                + self.phon_position_embedding.weight[None, : len(tokens[0])]
            )  # GE: independent positional encodings

    def embed(
        self, orthography, orthography_padding_mask, phonology, phonology_padding_mask
    ):
        orthography, phonology = self.embed_tokens(orthography, "o"), self.embed_tokens(
            phonology, "p"
        )

        orthography_encoding = self.orthography_encoder(
            orthography, src_key_padding_mask=orthography_padding_mask
        )
        #print(f"orthography_encoding.shape = {orthography_encoding.shape}")
        phonology_encoding = self.phonology_encoder(
            phonology, src_key_padding_mask=phonology_padding_mask
        )
        #print(f"phonology_encoding.shape = {phonology_encoding.shape}")
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
        #print(f"gp_encoding.shape = {gp_encoding.shape}")
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
        #print(f"pg_encoding.shape = {pg_encoding.shape}")

        # Concatenate outputs of cross-attention modules and add residual connection
        gp_pg = torch.cat((gp_encoding, pg_encoding), dim=1) + torch.cat(
            (orthography_encoding, phonology_encoding), dim=1
        )
        #print("gp_pg.shape = ", gp_pg.shape)
        # Concatenate padding masks
        gp_pg_padding_mask = torch.cat(
            (orthography_padding_mask, phonology_padding_mask), dim=-1
        )
        #print("gp_pg_padding_mask.shape = ", gp_pg_padding_mask.shape)

        global_embedding = self.global_embedding.repeat(gp_pg.shape[0], 1, 1)
        #print("global_embedding.shape = ", global_embedding.shape)
        gp_pg = torch.cat((global_embedding, gp_pg), dim=1)
        #print("gp_pg.shape = ", gp_pg.shape)
        gp_pg_padding_mask = torch.cat(
            (
                torch.zeros((gp_pg.shape[0], self.d_embedding),
                            device=gp_pg.device, dtype=torch.bool),
                gp_pg_padding_mask,
            ),
            dim=-1,
        )
        #print("gp_pg_padding_mask.shape = ", gp_pg_padding_mask.shape)

        mixed_encoding = self.transformer_mixer(
            gp_pg, src_key_padding_mask=gp_pg_padding_mask
        )
        #print("0 mixed_encoding.shape = ", mixed_encoding.shape)

        #final_encoding = (self.reduce(mixed_encoding[:, :self.d_embedding]).unsqueeze(-2) + global_embedding)
        #final_encoding = self.reduce_layer_norm(final_encoding)

        # Add a residual connection to the final encoding
        final_encoding = mixed_encoding[:,
                                        :self.d_embedding] + global_embedding
        #print("1 final_encoding.shape = ", final_encoding.shape)

        return final_encoding

    def forward(
        self,
        orth_enc_input,
        orth_enc_pad_mask,
        orth_dec_input,
        orth_dec_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        phon_dec_input,
        phon_dec_pad_mask,
    ):
        mixed_encoding = self.embed(
            orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
        )
        # print(mixed_encoding[0,0,:10])
        orth_dec_input = self.embed_tokens(orth_dec_input, "o")
        orth_ar_mask = self.generate_triangular_mask(
            orth_dec_input.shape[1], orth_dec_input.device
        )

        # print("Type of orth_dec_input = ", orth_dec_input.dtype)
        # print("Type of orth_ar_mask = ", orth_ar_mask.dtype)
        # print("Type of orth_pec_pad_mask = ", orth_dec_pad_mask.dtype)
        # print("Type of mixed_encoding = ", mixed_encoding.dtype)

        orth_output = self.orthography_decoder(
            tgt=orth_dec_input,
            tgt_mask=orth_ar_mask,
            tgt_key_padding_mask=orth_dec_pad_mask,
            memory=mixed_encoding,   
        )

        phon_dec_input = self.embed_tokens(phon_dec_input, "p")
        phon_ar_mask = self.generate_triangular_mask(
            phon_dec_input.shape[1], phon_dec_input.device
        )

        # print("Type of phon_dec_input = ", phon_dec_input.dtype)
        # print("Type of phon_ar_mask = ", phon_ar_mask.dtype)
        # print("Type of phon_pec_pad_mask = ", phon_dec_pad_mask.dtype)
        phon_output = self.phonology_decoder(
            tgt=phon_dec_input,
            tgt_mask=phon_ar_mask,
            tgt_key_padding_mask=phon_dec_pad_mask,
            memory=mixed_encoding,
        )
        # print(phon_output.shape,phon_output[0,0,:10],mixed_encoding[0,0,:10])
        B, OC, E = orth_output.shape
        # orth_output = orth_output.view(B*OC, E)
        orth_token_logits = self.linear_orthography_decoder(orth_output)
        B, PC, E = phon_output.shape
        # phon_output = phon_output.view(B*PC, E)
        phon_token_logits = self.linear_phonology_decoder(phon_output)
        # print("(B, C))
        # print("orth_token_logits.shape = ", orth_token_logits.shape)
        # print("phon_token_logits.shape = ", phon_token_logits.shape)

        orth_token_logits = orth_token_logits.transpose(1, 2)
        phon_token_logits = phon_token_logits.view(
            B, PC, 2, -1).transpose(1, 2)
        return {"orth": orth_token_logits, "phon": phon_token_logits}
        # return {'orth': orth_token_logits.view(B, self.vocab_sizes['orth_vocab_size'], OC),
        #        'phon': phon_token_logits.view(B, 2, PC, self.vocab_sizes['phon_vocab_size'] - 1)} # -1 because targets do not contain PAD

    def generate(
        self,
        orth_enc_input,
        orth_enc_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        deterministic=False,
    ):
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            # Output is mixed mode
            prompt_encoding = self.embed(
                orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask
            )
        # print(prompt_encoding[0,0,:10])
        mask = self.generate_triangular_mask(self.max_seq_len, device)
        # print("orth_enc_input = ", orth_enc_input)
        # print("max_seq_len = ", self.max_seq_len)
        # print("max_orth_seq_len = ", self.max_orth_seq_len)
        # print("max_phon_seq_len = ", self.max_phon_seq_len)
        # print("mask: ", mask)

        generated_orth_tokens = torch.tensor(
            [[0]], dtype=torch.long, device=device)
        generated_orth_embeddings = self.embed_tokens(
            generated_orth_tokens, "o")
        # print("generated_orth_embeddings = ", generated_orth_embeddings)

        generated_phon_tokens = [
            [torch.tensor([31], dtype=torch.long, device=device)]]
        generated_phon_embeddings = self.embed_tokens(
            generated_phon_tokens, "p")
        # print("generated_phon_embeddings = ", generated_phon_embeddings)
        # print(generated_phon_tokens, generated_phon_embeddings.shape,
        #       generated_phon_embeddings[0, 0, :10])

        # Iterate through the decoder only
        for step in range(self.max_seq_len - 1):
            # print("step = ", step)
            step_mask = mask[: step + 1, : step + 1]
            # print("step_mask = ", step_mask)

            with torch.no_grad():
                orth_output = self.orthography_decoder(
                    generated_orth_embeddings,
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                )
                B, OC, E = orth_output.shape
                # print("orth_output.shape = ", orth_output.shape)
                # orth_output = orth_output.view(B*OC, E)
                linear_output = self.linear_orthography_decoder(orth_output)
                # print("linear_output.shape = ", linear_output.shape)
                orthography_token_logits = linear_output.transpose(1, 2)
                # print("orthography_token_logits.shape = ", orthography_token_logits.shape)
                # orthography_token_logits = orthography_token_logits.view(B, self.vocab_sizes['orth_vocab_size'], OC)
                # print("orthography_token_logits = ", orthography_token_logits)

                phon_output = self.phonology_decoder(
                    generated_phon_embeddings,
                    memory=prompt_encoding,
                    tgt_mask=step_mask,
                )
                # print("phon_output.shape = ", phon_output.shape)
                # print(phon_output.shape,phon_output[0,0,:10],prompt_encoding[0,0,:10])
                B, PC, E = phon_output.shape
                # phon_output = phon_output.view(B*PC, E)
                phonology_token_logits = self.linear_phonology_decoder(
                    phon_output)
                # print("phonology_token_logits.shape = ", phonology_token_logits.shape)
                phonology_token_logits = phonology_token_logits.view(
                    B, PC, 2, -1
                ).transpose(1, 2)
                # print("phonology_token_logits.shape = ", phonology_token_logits.shape)
                # print("phonology_token_logits = ", phonology_token_logits)

                last_token_logits = (
                    orthography_token_logits[:, :, -1],
                    phonology_token_logits[:, :, -1, :],
                )
                # print(last_token_logits[1][:,1])

                # print("last_token_logits = ", last_token_logits)
                last_token_probs = (
                    torch.softmax(last_token_logits[0], dim=1),
                    torch.softmax(last_token_logits[1], dim=1),
                )
                # print("last_token_probs = ", last_token_probs)
                # print("last_token_probs[0].sum() = ", last_token_probs[0].sum())
                # print("last_token_probs[1].sum(dim=1) = ", last_token_probs[1].sum(dim=1))

                if deterministic:
                    new_orthography_token = last_token_probs[0].argmax(
                        dim=1, keepdim=True
                    )

                    new_phonology_vec = last_token_probs[1][:, 1] > 0.5
                    new_phonology_tokens = torch.where(new_phonology_vec)[1]
                else:
                    new_orthography_token = torch.multinomial(
                        last_token_probs[0], num_samples=1
                    )
                    # print("last_token_probs[1].shape = ", last_token_probs[1].shape)
                    # print("last_token_probs[1] = ", last_token_probs[1])
                    new_phonology_vec = torch.bernoulli(
                        last_token_probs[1][:, 1])
                    if new_phonology_vec.eq(0).all():
                        new_phonology_vec[0, 32] = 1
                    # print("new_phonology_vec.shape = ", new_phonology_vec.shape)
                    new_phonology_tokens = torch.where(new_phonology_vec)[
                        1
                    ]  # What happens if this returns all zeros?
                    # print("new_phonology_tokens.shape = ", new_phonology_tokens.shape)

                generated_orth_tokens = torch.cat(
                    (generated_orth_tokens, new_orthography_token), dim=-1
                )
                print("generated_orth_tokens = ", generated_orth_tokens)
                generated_orth_embeddings = self.embed_tokens(
                    generated_orth_tokens, "o"
                )
                # print("generated_orth_embeddings = ", generated_orth_embeddings)

                generated_phon_tokens[0].append(new_phonology_tokens)
                print("generated_phon_tokens = ", generated_phon_tokens)
                generated_phon_embeddings = self.embed_tokens(  # <<< ERROR
                    generated_phon_tokens, "p"
                )
                # print("generated_phon_embeddings = ", generated_phon_embeddings)

        return {"orth": generated_orth_tokens, "phon": generated_phon_tokens}

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
