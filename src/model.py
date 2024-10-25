import torch
from typing import List


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
        self.max_orth_seq_len = 100 #dataset.max_orth_seq_len
        self.max_phon_seq_len = 100 #dataset.max_phon_seq_len
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
        batch_size = len(tokens)
        # Every batch should be the same size. If this function is called from the forward routine, then the dataset.encode
        # routine will have already added the necessary padding. If this function is called from the generate routine, then
        # each successive phonological vector (list of active features) will have been generated at the same time. So we can
        # set the max_phon_len to the length of the first batch, since all batches should be the same length.
        max_phon_len = len(tokens[0])

        device = next(self.parameters()).device  # device of weights
        # len(tokens) is the batch size
        output_embedding = torch.zeros(
            (batch_size, max_phon_len, self.d_model), device=device
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

    def embed_p2p(self, phonology, phonology_padding_mask):
        return self.embed_p2o(phonology, phonology_padding_mask)

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
        elif pathway == "p2p":
            encoding = self.embed_p2p(phon_enc_input, phon_enc_pad_mask)
            phon_dec_input = self.embed_phon_tokens(phon_dec_input)
            phon_ar_mask = self.generate_triangular_mask(
                phon_dec_input.shape[1], phon_dec_input.device
            )
            phon_output = self.phonology_decoder(
                tgt=phon_dec_input,
                tgt_mask=phon_ar_mask,
                tgt_key_padding_mask=phon_dec_pad_mask,
                memory=encoding,
            )
            B, PC, E = phon_output.shape
            phon_token_logits = self.linear_phonology_decoder(phon_output)
            phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
            return {"phon": phon_token_logits}
            
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

    # Old
    def ortho_sample(self, last_token_probs, deterministic):
        if deterministic:
            token = last_token_probs.argmax(dim=1, keepdim=True)
        else:
            token = torch.multinomial(last_token_probs, num_samples=1)

        return token

    # Old
    def phono_sample(self, last_token_probs, deterministic):
        """
        Sample from phonological decoder output. last_token_probs is a tensor of shape (batch_size, 2, 33)
        where 2 represents the probability dimension and 33 is the number of possible phonological vector
        features (including BOS, EOS, PAD). For the probabilitye dimension (2) the zeroth index is the probability
        of the feature being off, and the first index is the probability of the feature being on.

        For example [0.6, 0.4] -> [feature off, feature on] and in this scenario the feature is off
        """

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

    # Old
    def orthography_decoder_loop(
        self,
        mask,
        generated_orth_embeddings,
        generated_orth_tokens,
        prompt_encoding,
        deterministic,
    ):
        # LOOP THROUGH ORTHOGRAPHY DECODER
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
        mask,
        generated_phon_embeddings,
        generated_phon_tokens: List[torch.Tensor],
        prompt_encoding,
        deterministic,
    ):
        # LOOP THROUGH PHONOLOGY DECODER.
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
        pathway,
        orth_enc_input,
        orth_enc_pad_mask,
        phon_enc_input,
        phon_enc_pad_mask,
        deterministic=False,
    ):
        """
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
        device = next(self.parameters()).device
        batch_size = orth_enc_input.shape[0]

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

        generated_phon_probs = None
        generated_phon_vecs = None
        generated_phon_tokens = None
        if pathway in ["op2op", "o2p"]:
            mask = self.generate_triangular_mask(self.max_phon_seq_len, device)

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
                [torch.tensor([31], dtype=torch.long, device=device)]
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
            mask = self.generate_triangular_mask(self.max_orth_seq_len, device)
            generated_orth_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
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
