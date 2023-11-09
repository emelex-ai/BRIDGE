import torch

class Encoder(torch.nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1):
        # Initialize the Encoder module
        super(Encoder, self).__init__()
        # Create a transformer encoder layer
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4*d_model)
        # Create the transformer encoder using the encoder layer
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Apply the transformer encoder to the input sequence
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class Decoder(torch.nn.Module):
    def __init__(self, d_model=512, nhead=1, num_layers=1):
        # Initialize the Decoder module
        super().__init__()
        # Create a transformer decoder layer
        decoder_layer = torch.nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=4*d_model)
        # Create the transformer decoder using the decoder layer
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Apply the transformer decoder to the target sequence with memory from the encoder
        output = self.transformer_decoder(tgt, memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        return output

class Model(torch.nn.Module):
    def __init__(self,
                 orth_vocab_size,
                 phon_vocab_size,
                 d_model=512,
                 nhead=1,
                 num_layers=1,
                 max_seq_len=20):
        # Initialize the main model
        super().__init__()

        # Initial embeddings for orthography, phonology, and position
        # Embedding for orthography
        self.orthography_embedding = torch.nn.Embedding(orth_vocab_size, d_model)
        self.orth_position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        # Embedding for phonology
        self.phonology_embedding = torch.nn.Embedding(phon_vocab_size, d_model)
        self.phon_position_embedding = torch.nn.Embedding(max_seq_len, d_model)
        
        # Global embedding parameter
        self.global_embedding = torch.nn.Parameter(torch.randn((1, 1, d_model))/d_model**0.5, requires_grad=True)

        # Encoder modules for orthography and phonology
        self.orthography_encoder = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.phonology_encoder = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers)

        # Cross-attention modules for orthography/phonology
        self.gp_multihead_attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.gp_layer_norm = torch.nn.LayerNorm(d_model)
        self.pg_multihead_attention = torch.nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        self.pg_layer_norm = torch.nn.LayerNorm(d_model)

        # Mixer and linear layer for transforming mixed encoding
        self.transformer_mixer = Encoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.reduce = torch.nn.Linear(d_model, d_model)
        self.reduce_layer_norm = torch.nn.LayerNorm(d_model)

        # Decoder modules for orthography and phonology
        self.orthography_decoder = Decoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.linear_orthography_decoder = torch.nn.Linear(d_model, orth_vocab_size)
        self.phonology_decoder = Decoder(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.linear_phonology_decoder = torch.nn.Linear(d_model, 2*(phon_vocab_size-1))

    def generate_triangular_mask(self, size, device):
        # Generate a size×size, strictly upper-triangular Boolean tensor
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), 1)
        return mask

    def embed_tokens(self, tokens, mode='o'):
        # Embed tokens for orthography or phonology
        assert mode in ['o','p']
        
        if mode == 'o':
            # Embedding for orthography
            assert isinstance(tokens, torch.Tensor), "For orthographic embeddings, tokens must be a pytorch tensor of integers"
            assert tokens.dtype == torch.long or tokens.dtype == torch.int, "Input tensor to Embedding must be type int or long"
            return self.orthography_embedding(tokens) + self.orth_position_embedding.weight[None, :tokens.shape[1]]
        else:
            # Embedding for phonology (averaging embeddings for each feature)
            assert isinstance(tokens, list), "For phonological vectors, tokens must be a list of pytorch tensors"
            for token in tokens:
                assert isinstance(token, torch.Tensor), "Each element in the list must be a pytorch tensor of integers"
            device = next(self.parameters()).device
            output_embedding = torch.zeros((len(tokens), len(tokens[0]), self.d_model), device=device)
            for batch_num, batch in enumerate(tokens):
                for indx, tokes in enumerate(batch):
                    avg_embedding = self.phonology_embedding(tokes).mean(axis=0)
                    output_embedding[batch_num, indx, :] = avg_embedding
            return output_embedding + self.phon_position_embedding.weight[None, :len(tokens[0])]

    def embed(self, orthography, orthography_padding_mask, phonology, phonology_padding_mask):
        # Embed orthography and phonology tokens and apply encoders
        orthography, phonology = self.embed_tokens(orthography, 'o'), self.embed_tokens(phonology, 'p')
        orthography_encoding = self.orthography_encoder(orthography, src_key_padding_mask=orthography_padding_mask)
        phonology_encoding = self.phonology_encoder(phonology, src_key_padding_mask=phonology_padding_mask)

        # Apply cross-attention modules
        gp_encoding = self.gp_multihead_attention(orthography_encoding, phonology_encoding, phonology_encoding,
                                                  key_padding_mask=phonology_padding_mask)[0] + orthography_encoding
        gp_encoding = self.gp_layer_norm(gp_encoding)
        pg_encoding = self.pg_multihead_attention(phonology_encoding, orthography_encoding, orthography_encoding,
                                                  key_padding_mask=orthography_padding_mask)[0] + phonology_encoding
        pg_encoding = self.pg_layer_norm(pg_encoding)

        # Concatenate outputs of cross-attention modules
        gp_pg = torch.cat((gp_encoding, pg_encoding), dim=1) + torch.cat((orthography_encoding, phonology_encoding), dim=1)

        # Concatenate padding masks
        gp_pg_padding_mask = torch.cat((orthography_padding_mask, phonology_padding_mask), dim=-1)

        # Apply global embedding
        global_embedding = self.global_embedding.repeat(gp_pg.shape[0], 1, 1)
        gp_pg = torch.cat((global_embedding, gp_pg), dim=1)
        gp_pg_padding_mask = torch.cat((torch.zeros((gp_pg.shape[0], 1), device=gp_pg.device, dtype=torch.bool), gp_pg_padding_mask), dim=-1)

        # Mix representations
        mixed_encoding = self.transformer_mixer(gp_pg, src_key_padding_mask=gp_pg_padding_mask)

        # Reduce dimensionality
        final_encoding = self.reduce(mixed_encoding[:, 0]).unsqueeze(-2) + global_embedding
        final_encoding = self.reduce_layer_norm(final_encoding)

        return final_encoding

    def forward(self, orth_enc_input, orth_enc_pad_mask, orth_dec_input, orth_dec_pad_mask,
                      phon_enc_input, phon_enc_pad_mask, phon_dec_input, phon_dec_pad_mask):
        # Perform the forward pass of the model
        mixed_encoding = self.embed(orth_enc_input, orth_enc_pad_mask,
                                    phon_enc_input, phon_enc_pad_mask)
        orth_dec_input = self.embed_tokens(orth_dec_input, 'o')
        orth_ar_mask = self.generate_triangular_mask(orth_dec_input.shape[1], orth_dec_input.device)

        orth_output = self.orthography_decoder(tgt=orth_dec_input,
                                               tgt_mask=orth_ar_mask,
                                               tgt_key_padding_mask=orth_dec_pad_mask,
                                               memory=mixed_encoding)

        phon_dec_input = self.embed_tokens(phon_dec_input, 'p')
        phon_ar_mask = self.generate_triangular_mask(phon_dec_input.shape[1], phon_dec_input.device)

        phon_output = self.phonology_decoder(tgt=phon_dec_input,
                                              tgt_mask=phon_ar_mask,
                                              tgt_key_padding_mask=phon_dec_pad_mask,
                                              memory=mixed_encoding)

        B, OC, E = orth_output.shape
        orth_token_logits = self.linear_orthography_decoder(orth_output)
        B, PC, E = phon_output.shape
        phon_token_logits = self.linear_phonology_decoder(phon_output)

        orth_token_logits = orth_token_logits.transpose(1, 2)
        phon_token_logits = phon_token_logits.view(B, PC, 2, -1).transpose(1, 2)
        return {'orth': orth_token_logits, 'phon': phon_token_logits}

    def generate(self, orth_enc_input, orth_enc_pad_mask, phon_enc_input, phon_enc_pad_mask, deterministic=False):
        # Generate sequences using the trained model
        self.eval()
        device = next(self.parameters()).device

        with torch.no_grad():
            prompt_encoding = self.embed(orth_enc_input,
                                         orth_enc_pad_mask,
                                         phon_enc_input,
                                         phon_enc_pad_mask)

        mask = self.generate_triangular_mask(self.max_seq_len, device)

        generated_orth_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
        generated_orth_embeddings = self.embed_tokens(generated_orth_tokens, 'o')

        generated_phon_tokens = [[torch.tensor([31], dtype=torch.long, device=device)]]
        generated_phon_embeddings = self.embed_tokens(generated_phon_tokens, 'p')

        for step in range(self.max_seq_len-1):
            step_mask = mask[:step+1, :step+1]

            with torch.no_grad():
                orth_output = self.orthography_decoder(generated_orth_embeddings, memory=prompt_encoding, tgt_mask=step_mask)
                B, OC, E = orth_output.shape
                linear_output = self.linear_orthography_decoder(orth_output)
                orthography_token_logits = linear_output.transpose(1, 2)

                phon_output = self.phonology_decoder(generated_phon_embeddings, memory=prompt_encoding, tgt_mask=step_mask)
                B, PC, E = phon_output.shape
                phonology_token_logits = self.linear_phonology_decoder(phon_output)
                phonology_token_logits = phonology_token_logits.view(B, PC, 2, -1).transpose(1, 2)

                last_token_logits = (orthography_token_logits[:, :, -1], phonology_token_logits[:, :, -1, :])

                last_token_probs = (
                    torch.softmax(last_token_logits[0], dim=1),
                    torch.softmax(last_token_logits[1], dim=1)
                )

                if deterministic:
                    new_orthography_token = last_token_probs[0].argmax(dim=1, keepdim=True)
                    new_phonology_vec = last_token_probs[1][:, 1] > .5
                    new_phonology_tokens = torch.where(new_phonology_vec)[1]
                else:
                    new_orthography_token = torch.multinomial(last_token_probs[0], num_samples=1)
                    new_phonology_vec = torch.bernoulli(last_token_probs[1][:, 1])
                    if new_phonology_vec.eq(0).all():
                        new_phonology_vec[0, 32] = 1
                    new_phonology_tokens = torch.where(new_phonology_vec)[1]

                generated_orth_tokens = torch.cat((generated_orth_tokens, new_orthography_token), dim=-1)
                generated_orth_embeddings = self.embed_tokens(generated_orth_tokens, 'o')
                generated_phon_tokens[0].append(new_phonology_tokens)
                generated_phon_embeddings = self.embed_tokens(generated_phon_tokens, 'p')

        return {'orth': generated_orth_tokens, 'phon': generated_phon_tokens}

    def size(self):
        # Calculate and return the number of parameters and model size
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
        result = {'parameters': param_num_all,
                  'size in MB': size_all_mb}
        return result
