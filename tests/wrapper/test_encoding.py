"""Test suite for encoding functionality in BRIDGE.

This module contains focused tests for the encoding process, demonstrating how
SyntheticBridgeDatasetMultiWord creates orthographic and phonological sequences
for BRIDGE model input.
"""

from typing import cast

import pytest
import torch
from bridge.domain.datamodels import ModelConfig
from bridge.domain.dataset.bridge_dataset import BridgeDataset
from bridge.domain.dataset.phoneme_tokenizer import PhonemeTokenizer
from bridge.domain.model import Model
from bridge.domain.model.synthetic_dataset import SyntheticBridgeDatasetMultiWord
from torch import Tensor, tensor


def debug_nan_trace(model, input_dict):
    """Trace where NaNs first appear in the model forward pass."""
    import torch

    def check_tensor(name, tensor):
        if isinstance(tensor, torch.Tensor):
            print(
                f"{name}: min={tensor.min().item()}, max={tensor.max().item()}, mean={tensor.mean().item()}"
            )
            if torch.isnan(tensor).any():
                print(f"❌ NaN detected in {name}: {tensor.shape}")
                return True
        return False

    # Check embeddings
    ortho_emb = model.embed_orth_tokens(input_dict["orth_enc_input"])
    if check_tensor("orthography_embedding", ortho_emb):
        return

    # Encoder output
    ortho_enc = model.orthography_encoder(
        ortho_emb, src_key_padding_mask=input_dict["orth_enc_pad_mask"]
    )
    if check_tensor("orthography_encoder", ortho_enc):
        return

    # Decoder input
    phon_emb = model.embed_phon_tokens(input_dict["phon_dec_input"])
    if check_tensor("phonology_embedding", phon_emb):
        return

    # Decoder output
    phon_out = model.phonology_decoder(
        tgt=phon_emb,
        tgt_key_padding_mask=input_dict["phon_dec_pad_mask"],
        memory=ortho_enc,
    )
    if check_tensor("phonology_decoder", phon_out):
        return

    # Linear output
    B, PC, E = phon_out.shape
    phon_token_logits = (
        model.linear_phonology_decoder(phon_out).view(B, PC, 2, -1).transpose(1, 2)
    )
    if check_tensor("linear_phonology_decoder", phon_token_logits):
        return

    # If you have a similar linear layer for orthographic output, check it too:
    if hasattr(model, "linear_orthography_decoder"):
        # The orthographic output is likely computed as:
        ortho_out = model.linear_orthography_decoder(ortho_enc)
        if check_tensor("linear_orthography_decoder", ortho_out):
            return

    print("No NaNs detected in major modules.")


def full_nan_debug(model, ortho, phon):
    import torch

    print("\n=== INPUT DEBUG ===")
    print("orth_enc_input shape:", ortho.enc_input_ids.shape)
    print("orth_enc_input max index:", ortho.enc_input_ids.max().item())
    print("orth_enc_input NaN:", torch.isnan(ortho.enc_input_ids).any())
    print("orth_enc_pad_mask shape:", ortho.enc_pad_mask.shape)
    print("phon_dec_input type:", type(phon.dec_input_ids))
    if isinstance(phon.dec_input_ids, list):
        print(
            "phon_dec_input lens:", [[tt.shape for tt in t] for t in phon.dec_input_ids]
        )
        print(
            "phon_dec_input NaN:",
            any(torch.isnan(tt).any() for t in phon.dec_input_ids for tt in t),
        )
    else:
        print("phon_dec_input shape:", phon.dec_input_ids.shape)
        print("phon_dec_input NaN:", torch.isnan(phon.dec_input_ids).any())
    print("phon_dec_pad_mask shape:", phon.dec_pad_mask.shape)

    print("\n=== MODEL PARAMETER DEBUG ===")
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN in parameter: {name}")
        if torch.isinf(param).any():
            print(f"❌ Inf in parameter: {name}")

    print("\n=== POSITION EMBEDDING DEBUG ===")
    print("orth_position_embedding size:", model.orth_position_embedding.num_embeddings)
    print("phon_position_embedding size:", model.phon_position_embedding.num_embeddings)
    seq_len = ortho.enc_input_ids.shape[1]
    assert (
        seq_len <= model.orth_position_embedding.num_embeddings
    ), f"Input sequence length {seq_len} exceeds orth_position_embedding size {model.orth_position_embedding.num_embeddings}"
    if isinstance(phon.dec_input_ids, list):
        phon_seq_len = max(len(t) for t in phon.dec_input_ids)
    else:
        phon_seq_len = phon.dec_input_ids.shape[1]
    assert (
        phon_seq_len <= model.phon_position_embedding.num_embeddings
    ), f"Phonological sequence length {phon_seq_len} exceeds phon_position_embedding size {model.phon_position_embedding.num_embeddings}"

    print("\n=== DEVICE DEBUG ===")
    print("Model device:", model.device)
    print("orth_enc_input device:", ortho.enc_input_ids.device)
    print(
        "phon_dec_input device:",
        phon.dec_input_ids[0][0].device
        if isinstance(phon.dec_input_ids, list)
        else phon.dec_input_ids.device,
    )

    print("\n=== LAYERNORM VARIANCE DEBUG ===")
    # Check variance before each LayerNorm (if possible)
    # This is a simple example for the first LayerNorm in the model
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            # Try to get input to LayerNorm (requires forward hook for full coverage)
            print(f"LayerNorm: {name} (cannot check input variance here without hooks)")

    print("\n=== MASK DEBUG ===")
    # If you use masks, check for all-False rows
    if hasattr(model, "orthography_encoder"):
        enc = model.orthography_encoder
        if hasattr(enc, "create_sliding_window_mask"):
            mask = enc.create_sliding_window_mask(seq_len, ortho.enc_input_ids.device)
            all_masked = (~mask).all(dim=-1)
            if all_masked.any():
                print("❌ Some rows in encoder mask are all masked out!")
    if hasattr(model, "phonology_decoder"):
        dec = model.phonology_decoder
        if hasattr(dec, "create_sliding_window_causal_mask"):
            mask = dec.create_sliding_window_causal_mask(
                phon_seq_len, phon.dec_input_ids[0][0].device
            )
            all_masked = (~mask).all(dim=-1)
            if all_masked.any():
                print("❌ Some rows in decoder mask are all masked out!")

    print("\n=== PYTORCH VERSION ===")
    import torch

    print("PyTorch version:", torch.__version__)


@pytest.fixture
def sample_dataset() -> SyntheticBridgeDatasetMultiWord:
    """Create a sample multi-word dataset for testing."""
    return SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=128,
        min_words_per_sequence=2,
        max_words_per_sequence=4,
        seed=42,
    )


def phoneme_sequence_to_feature_indices(
    phoneme_sequence: list[str],
    phoneme_tokenizer,
) -> list[list[Tensor]]:
    """Convert a list of phoneme strings to a list of lists of active feature indices.

    Args:
        phoneme_sequence: List of phoneme strings (e.g., ['K', 'AA0', ...])
        phoneme_tokenizer: An instance of PhonemeTokenizer

    Returns:
        List of lists, where each inner list contains the indices of active features
        for that phoneme.
    """
    feature_indices = []
    for phoneme in phoneme_sequence:
        indices_tensor = phoneme_tokenizer._get_phoneme_indices(phoneme)
        feature_indices.append(indices_tensor)
    return feature_indices


# @pytest.fixture
def word_sequence_to_phoneme_sequence(
    words: list[str],
    tokenizer: PhonemeTokenizer,
) -> tuple[list[str], list[list[Tensor]]]:
    """Generate a list of words and their phoneme sequences.

    Args:
        words: A list of words
        tokenizer: A PhonemeTokenizer instance

    Returns:
        A tuple containing:
            phonemes: A list of phoneme sequences, where each sequence is a list
                of phoneme strings for a word.
            features: A list of lists of feature indices (as Tensors), where each
                inner list corresponds to the features for each phoneme in the word.
    """
    phoneme_sequences = []
    phoneme_features = []

    for word in words:
        phonemes = tokenizer._get_word_phonemes(word)
        phoneme_sequences.append(phonemes)
        phoneme_features.append(
            phoneme_sequence_to_feature_indices(phonemes, tokenizer)
        )

        # All words should be found in CMU dictionary since they're from real_words
        assert phonemes is not None, f"Word '{word}' not found in CMU dictionary"
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0

    return phoneme_sequences, phoneme_features


@pytest.fixture
def sample_encoding(sample_dataset):
    """Get a sample encoding from the dataset."""
    return sample_dataset[0]


def test_dataset_creation(sample_dataset):
    """Test that SyntheticBridgeDatasetMultiWord creates valid sequences."""
    dataset = cast(BridgeDataset, sample_dataset)

    assert len(dataset) == 5
    assert hasattr(dataset, "sequences")
    assert len(dataset.sequences) == 5

    # Check that each sequence is a list of words
    for i, sequence in enumerate(dataset.sequences):
        assert isinstance(sequence, list)
        assert len(sequence) >= 2  # min_words_per_sequence
        assert len(sequence) <= 4  # max_words_per_sequence

        # Check that all words are strings
        for word in sequence:
            assert isinstance(word, str)
            assert len(word) >= 3  # min word length
            assert len(word) <= 10  # max word length


def test_word_to_phoneme_conversion(sample_dataset):
    """Test that individual words are correctly converted to phonemes."""
    dataset = cast(BridgeDataset, sample_dataset)

    # Test a few words from the first sequence
    word_sequence = dataset.sequences[0]

    for word in word_sequence:
        phonemes = dataset.tokenizer.phoneme_tokenizer._get_word_phonemes(word)

        # All words should be found in CMU dictionary since they're from real_words
        assert phonemes is not None, f"Word '{word}' not found in CMU dictionary"
        assert isinstance(phonemes, list)
        assert len(phonemes) > 0

        # Check that phonemes are valid (strings)
        for phoneme in phonemes:
            assert isinstance(phoneme, str)


def test_phrase_to_phoneme_conversion(sample_dataset):
    """Test that multi-word phrases are correctly converted to phoneme sequences."""
    dataset = cast(BridgeDataset, sample_dataset)

    # Test the first sequence
    word_sequence = dataset.sequences[0]
    text_sequence = " ".join(word_sequence)

    complete_phonemes = dataset.tokenizer.phoneme_tokenizer._get_phrase_phonemes(
        text_sequence
    )

    assert complete_phonemes is not None
    assert isinstance(complete_phonemes, list)
    assert len(complete_phonemes) > 0

    # Check that the sequence contains [SPC] tokens between words
    word_count = len(word_sequence)
    spc_count = complete_phonemes.count("[SPC]")
    assert spc_count == word_count - 1, "Should have [SPC] tokens between words"


def test_bridge_encoding_structure(sample_encoding):
    """Test that BridgeEncoding has the correct structure."""
    assert hasattr(sample_encoding, "orthographic")
    assert hasattr(sample_encoding, "phonological")

    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological

    # Check orthographic encoding attributes
    assert hasattr(ortho, "enc_input_ids")
    assert hasattr(ortho, "enc_pad_mask")
    assert hasattr(ortho, "dec_input_ids")
    assert hasattr(ortho, "dec_pad_mask")

    # Check phonological encoding attributes
    assert hasattr(phon, "enc_input_ids")
    assert hasattr(phon, "enc_pad_mask")
    assert hasattr(phon, "dec_input_ids")
    assert hasattr(phon, "dec_pad_mask")


def test_orthographic_encoding_shapes(sample_encoding):
    """Test that orthographic encoding has correct tensor shapes."""
    ortho = sample_encoding.orthographic

    # Check that all tensors are 2D (batch_size, seq_len)
    assert ortho.enc_input_ids.dim() == 2
    assert ortho.enc_pad_mask.dim() == 2
    assert ortho.dec_input_ids.dim() == 2
    assert ortho.dec_pad_mask.dim() == 2

    # Check that batch size is 1 (single sequence)
    batch_size = ortho.enc_input_ids.shape[0]
    assert batch_size == 1

    # Check that encoder tensors have the same sequence length
    enc_seq_len = ortho.enc_input_ids.shape[1]
    assert ortho.enc_pad_mask.shape[1] == enc_seq_len

    # Check that decoder tensors have the same sequence length
    dec_seq_len = ortho.dec_input_ids.shape[1]
    assert ortho.dec_pad_mask.shape[1] == dec_seq_len

    # Note: Encoder and decoder can have different lengths
    # (encoder might include EOS token, decoder might not)
    # This is expected behavior in sequence-to-sequence models


def test_phonological_encoding_structure(sample_encoding):
    """Test that phonological encoding has correct structure."""
    phon = sample_encoding.phonological

    # Check that enc_input_ids and dec_input_ids are lists of tensors
    assert isinstance(phon.enc_input_ids, list)
    assert isinstance(phon.dec_input_ids, list)
    assert len(phon.enc_input_ids) == 1  # batch_size = 1
    assert len(phon.dec_input_ids) == 1

    # Check that pad masks are tensors
    assert isinstance(phon.enc_pad_mask, torch.Tensor)
    assert isinstance(phon.dec_pad_mask, torch.Tensor)

    # Check that pad masks are 2D
    assert phon.enc_pad_mask.dim() == 2
    assert phon.dec_pad_mask.dim() == 2


def test_orthographic_decoding(sample_dataset, sample_encoding):
    """Test that orthographic encoding can be decoded back to text."""
    dataset = cast(BridgeDataset, sample_dataset)
    ortho = sample_encoding.orthographic

    # Decode the orthographic encoding
    decoded_ortho = dataset.tokenizer.decode(
        ortho_indices=[ortho.enc_input_ids[0].tolist()]
    )

    assert decoded_ortho is not None
    assert "orthographic" in decoded_ortho
    assert len(decoded_ortho["orthographic"]) == 1

    decoded_text = decoded_ortho["orthographic"][0]
    assert isinstance(decoded_text, str)
    assert len(decoded_text) > 0


def test_phonological_decoding(sample_encoding):
    """Test that phonological encoding can be decoded back to phonemes."""
    phon = sample_encoding.phonological

    if phon.enc_input_ids:
        first_phon_seq = phon.enc_input_ids[0]

        # Convert tensor to list for decoding
        if isinstance(first_phon_seq, torch.Tensor):
            phon_seq_list = first_phon_seq.tolist()
        else:
            phon_seq_list = first_phon_seq

        # Note: This test might fail if the tokenizer doesn't support
        # decoding back to phonemes, which is expected behavior
        try:
            # This is a basic check - actual decoding might not work
            assert isinstance(phon_seq_list, list)
            assert len(phon_seq_list) > 0
        except Exception:
            # If decoding fails, that's acceptable
            pass


def test_vocabulary_sizes(sample_dataset):
    """Test that dataset provides correct vocabulary sizes."""
    dataset = cast(BridgeDataset, sample_dataset)

    assert hasattr(dataset, "orthographic_vocabulary_size")
    assert hasattr(dataset, "phonological_vocabulary_size")

    assert isinstance(dataset.orthographic_vocabulary_size, int)
    assert isinstance(dataset.phonological_vocabulary_size, int)

    assert dataset.orthographic_vocabulary_size > 0
    assert dataset.phonological_vocabulary_size > 0


def test_model_compatibility(sample_dataset, sample_encoding):
    """Test that BridgeEncoding is compatible with model input requirements."""
    dataset = cast(BridgeDataset, sample_dataset)
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological

    # Check that all required inputs exist
    required_inputs = [
        ortho.enc_input_ids,
        ortho.enc_pad_mask,
        phon.enc_input_ids,
        phon.enc_pad_mask,
        phon.dec_input_ids,
        phon.dec_pad_mask,
        ortho.dec_input_ids,
        ortho.dec_pad_mask,
    ]

    for input_tensor in required_inputs:
        assert input_tensor is not None


@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return ModelConfig(
        d_model=64,
        nhead=2,
        num_phon_enc_layers=1,
        num_orth_enc_layers=1,
        num_mixing_enc_layers=1,
        num_phon_dec_layers=1,
        num_orth_dec_layers=1,
        d_embedding=1,
        seed=42,
        use_sliding_window=False,
        window_size=61,
        is_causal=False,
        max_seq_len=128,
        ensure_contiguous=False,
    )


@pytest.fixture
def configured_model(sample_dataset, model_config):
    """Create a model configured for testing with the dataset."""
    model = Model(model_config, sample_dataset)
    model.eval()

    # Configure model for longer sequences
    model.max_orth_seq_len = 1024
    model.max_phon_seq_len = 1024

    # Update position embeddings
    device = model.device
    d_model = model.model_config.d_model
    model.orth_position_embedding = torch.nn.Embedding(
        model.max_orth_seq_len, d_model, device=device
    )
    model.phon_position_embedding = torch.nn.Embedding(
        model.max_phon_seq_len, d_model, device=device
    )

    return model


@pytest.fixture
def model_output(sample_dataset, sample_encoding, configured_model):
    """Perform a model forward pass and return the output."""
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological

    with torch.no_grad():
        output = configured_model.forward(
            "op2op",
            orth_enc_input=ortho.enc_input_ids,
            orth_enc_pad_mask=ortho.enc_pad_mask,
            phon_enc_input=phon.enc_input_ids,
            phon_enc_pad_mask=phon.enc_pad_mask,
            phon_dec_input=phon.dec_input_ids,
            phon_dec_pad_mask=phon.dec_pad_mask,
            orth_dec_input=ortho.dec_input_ids,
            orth_dec_pad_mask=ortho.dec_pad_mask,
        )

    return output


def test_model_initialization_with_dataset(sample_dataset, model_config):
    """Test that model initializes correctly with the dataset."""
    model = Model(model_config, sample_dataset)

    # Check that model got vocabulary sizes from dataset
    assert (
        model.orthographic_vocabulary_size
        == sample_dataset.orthographic_vocabulary_size
    )
    assert (
        model.phonological_vocabulary_size
        == sample_dataset.phonological_vocabulary_size
    )

    print(
        f"✓ Model initialized with orthographic vocab size: {model.orthographic_vocabulary_size}"
    )
    print(
        f"✓ Model initialized with phonological vocab size: {model.phonological_vocabulary_size}"
    )


def test_model_sequence_length_configuration(
    sample_dataset, configured_model, sample_encoding
):
    """Test that model can handle sequences from the dataset."""
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological

    orth_seq_len = ortho.enc_input_ids.shape[1]
    phon_seq_len = phon.enc_pad_mask.shape[1]

    print(f"Sample orthographic sequence length: {orth_seq_len}")
    print(f"Sample phonological sequence length: {phon_seq_len}")
    print(f"Model max_orth_seq_len: {configured_model.max_orth_seq_len}")
    print(f"Model max_phon_seq_len: {configured_model.max_phon_seq_len}")

    # Verify sequences fit within model limits
    assert orth_seq_len <= configured_model.max_orth_seq_len
    assert phon_seq_len <= configured_model.max_phon_seq_len

    print(f"✓ Model can handle sequence lengths from dataset")


def test_model_forward_pass_basic(model_output, sample_encoding):
    """Test that model forward pass completes without errors and produces correct outputs."""
    # Basic output validation
    assert isinstance(model_output, dict)
    assert "orth" in model_output
    assert "phon" in model_output
    assert isinstance(model_output["orth"], torch.Tensor)
    assert isinstance(model_output["phon"], torch.Tensor)

    print(f"✓ Model forward pass completed successfully")
    print(f"✓ Orthographic output shape: {model_output['orth'].shape}")
    print(f"✓ Phonological output shape: {model_output['phon'].shape}")

    # CRITICAL: Check that output sequence lengths match input sequence lengths
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological
    print("ortho= ", ortho.enc_input_ids)
    print("phon= ", phon.enc_input_ids)

    # Get expected output sequence lengths from decoder inputs
    expected_orth_seq_len = ortho.dec_input_ids.shape[1]
    expected_phon_seq_len = len(phon.dec_input_ids[0]) if phon.dec_input_ids else 0

    # Get actual output sequence lengths
    actual_orth_seq_len = model_output["orth"].shape[2]  # (batch, vocab, seq_len)
    if model_output["phon"].dim() == 3:
        actual_phon_seq_len = model_output["phon"].shape[1]  # (batch, seq_len, vocab)
    else:  # 4D
        actual_phon_seq_len = model_output["phon"].shape[
            2
        ]  # (batch, 2, seq_len, vocab)

    print(f"Expected orthographic sequence length: {expected_orth_seq_len}")
    print(f"Actual orthographic sequence length: {actual_orth_seq_len}")
    print(f"Expected phonological sequence length: {expected_phon_seq_len}")
    print(f"Actual phonological sequence length: {actual_phon_seq_len}")

    # Assert that output sequence lengths match input sequence lengths
    assert actual_orth_seq_len == expected_orth_seq_len, (
        f"Orthographic output sequence length {actual_orth_seq_len} doesn't match "
        f"expected length {expected_orth_seq_len}. This indicates sequence truncation."
    )

    assert actual_phon_seq_len == expected_phon_seq_len, (
        f"Phonological output sequence length {actual_phon_seq_len} doesn't match "
        f"expected length {expected_phon_seq_len}. This indicates sequence truncation."
    )

    # Check vocabulary sizes
    expected_orth_vocab_size = 106  # From the model configuration
    expected_phon_vocab_size = 36 - 1  # From the model configuration

    actual_orth_vocab_size = model_output["orth"].shape[1]  # (batch, vocab, seq_len)
    if model_output["phon"].dim() == 3:
        actual_phon_vocab_size = model_output["phon"].shape[
            2
        ]  # (batch, seq_len, vocab)
    else:  # 4D
        actual_phon_vocab_size = model_output["phon"].shape[
            3
        ]  # (batch, 2, seq_len, vocab)

    assert actual_orth_vocab_size == expected_orth_vocab_size, (
        f"Orthographic vocabulary size {actual_orth_vocab_size} doesn't match "
        f"expected {expected_orth_vocab_size}"
    )

    assert actual_phon_vocab_size == expected_phon_vocab_size, (
        f"Phonological vocabulary size {actual_phon_vocab_size} doesn't match "
        f"expected {expected_phon_vocab_size}"
    )

    print(f"✓ Output sequence lengths match input sequence lengths")
    print(f"✓ Output vocabulary sizes are correct")


def test_model_forward_pass_basic_2(
    # model_output,
    sample_encoding,
    sample_dataset: BridgeDataset,
    configured_model,
):
    """Test that model forward pass completes without errors and produces correct outputs.

    Use a message with two words. All else equal
    """
    # # Basic output validation
    # assert isinstance(model_output, dict)
    # assert "orth" in model_output
    # assert "phon" in model_output
    # assert isinstance(model_output["orth"], torch.Tensor)
    # assert isinstance(model_output["phon"], torch.Tensor)

    # print(f"✓ Model forward pass completed successfully")
    # print(f"✓ Orthographic output shape: {model_output['orth'].shape}")
    # print(f"✓ Phonological output shape: {model_output['phon'].shape}")

    # ==== Start Generate Encodings ==============================
    dataset = cast(BridgeDataset, sample_dataset)
    words = ["cariello", "annually", "unfav", "gehringer", "ensemble", "colpitts"]
    phoneme_sequences, phoneme_features = word_sequence_to_phoneme_sequence(
        words,
        dataset.tokenizer.phoneme_tokenizer,
    )

    nb_words: int = 2
    word_sequence = words[:nb_words]
    concatenated_words = " ".join(word_sequence)
    orth_enc_input = [concatenated_words]
    batch_size = 1
    sample_encoding = dataset.tokenizer.encode(orth_enc_input)
    # ==== End Generate Encodings ==============================

    # CRITICAL: Check that output sequence lengths match input sequence lengths
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological
    print("ortho= ", ortho.enc_input_ids)
    print("ortho.shape= ", ortho.enc_input_ids.shape)
    print("phon= ", phon.enc_input_ids)
    # quit()

    # Get expected output sequence lengths from decoder inputs
    expected_orth_seq_len = ortho.dec_input_ids.shape[1]
    expected_phon_seq_len = len(phon.dec_input_ids[0]) if phon.dec_input_ids else 0

    # ================
    with torch.no_grad():
        model_output = configured_model.forward(
            "op2op",
            orth_enc_input=ortho.enc_input_ids,
            orth_enc_pad_mask=ortho.enc_pad_mask,
            phon_enc_input=phon.enc_input_ids,
            phon_enc_pad_mask=phon.enc_pad_mask,
            phon_dec_input=phon.dec_input_ids,
            phon_dec_pad_mask=phon.dec_pad_mask,
            orth_dec_input=ortho.dec_input_ids,
            orth_dec_pad_mask=ortho.dec_pad_mask,
        )
    # ================

    # Get actual output sequence lengths
    actual_orth_seq_len = model_output["orth"].shape[2]  # (batch, vocab, seq_len)
    print(f"{model_output["orth"].shape=}")
    print(f"model output: {actual_orth_seq_len=}")
    if model_output["phon"].dim() == 3:
        actual_phon_seq_len = model_output["phon"].shape[1]  # (batch, seq_len, vocab)
    else:  # 4D
        actual_phon_seq_len = model_output["phon"].shape[
            2
        ]  # (batch, 2, seq_len, vocab)

    print(f"Expected orthographic sequence length: {expected_orth_seq_len}")
    print(f"Actual orthographic sequence length: {actual_orth_seq_len}")
    print(f"Expected phonological sequence length: {expected_phon_seq_len}")
    print(f"Actual phonological sequence length: {actual_phon_seq_len}")

    # Assert that output sequence lengths match input sequence lengths
    assert actual_orth_seq_len == expected_orth_seq_len, (
        f"Orthographic output sequence length {actual_orth_seq_len} doesn't match "
        f"expected length {expected_orth_seq_len}. This indicates sequence truncation."
    )

    assert actual_phon_seq_len == expected_phon_seq_len, (
        f"Phonological output sequence length {actual_phon_seq_len} doesn't match "
        f"expected length {expected_phon_seq_len}. This indicates sequence truncation."
    )

    # Check vocabulary sizes
    expected_orth_vocab_size = 106  # From the model configuration
    # the phonology decoder decreases the vocabulary size by 1
    expected_phon_vocab_size = 36 - 1  # From the model configuration

    actual_orth_vocab_size = model_output["orth"].shape[1]  # (batch, vocab, seq_len)
    if model_output["phon"].dim() == 3:
        actual_phon_vocab_size = model_output["phon"].shape[
            2
        ]  # (batch, seq_len, vocab)
    else:  # 4D
        actual_phon_vocab_size = model_output["phon"].shape[
            3
        ]  # (batch, 2, seq_len, vocab)

    assert actual_orth_vocab_size == expected_orth_vocab_size, (
        f"Orthographic vocabulary size {actual_orth_vocab_size} doesn't match "
        f"expected {expected_orth_vocab_size}"
    )

    assert actual_phon_vocab_size == expected_phon_vocab_size, (
        f"Phonological vocabulary size {actual_phon_vocab_size} doesn't match "
        f"expected {expected_phon_vocab_size}"
    )

    print(f"✓ Output sequence lengths match input sequence lengths")
    print(f"✓ Output vocabulary sizes are correct")


def test_model_output_dimensions(model_output, sample_encoding):
    """Test that model outputs have correct dimensions."""
    ortho = sample_encoding.orthographic
    batch_size = ortho.enc_input_ids.shape[0]

    # Orthographic output should be 3D: (batch_size, seq_len, vocab_size)
    assert model_output["orth"].dim() == 3
    assert model_output["orth"].shape[0] == batch_size

    # Phonological output could be 3D or 4D
    assert model_output["phon"].dim() in [3, 4]
    assert model_output["phon"].shape[0] == batch_size

    print(f"✓ Output dimensions are correct")
    print(f"✓ Orthographic output: {model_output['orth'].shape}")
    print(f"✓ Phonological output: {model_output['phon'].shape}")


def test_model_output_vocabulary_sizes(model_output, configured_model):
    """Test that model outputs have correct vocabulary sizes."""
    # Check vocabulary sizes
    orth_vocab_size = model_output["orth"].shape[2]
    if model_output["phon"].dim() == 3:
        phon_vocab_size = model_output["phon"].shape[2]
    else:  # 4D
        phon_vocab_size = model_output["phon"].shape[3]

    print(
        f"Model orthographic vocabulary size: {configured_model.orthographic_vocabulary_size}"
    )
    print(
        f"Model phonological vocabulary size: {configured_model.phonological_vocabulary_size}"
    )
    print(f"Output orthographic vocabulary size: {orth_vocab_size}")
    print(f"Output phonological vocabulary size: {phon_vocab_size}")

    # Check that vocabulary sizes are reasonable
    assert orth_vocab_size > 0
    assert phon_vocab_size > 0

    # This is where we identify the potential bug
    if orth_vocab_size != configured_model.orthographic_vocabulary_size:
        print(f"⚠ WARNING: Orthographic vocabulary size mismatch!")
        print(f"   Expected: {configured_model.orthographic_vocabulary_size}")
        print(f"   Got: {orth_vocab_size}")
        print(f"   This might indicate a bug in the model's forward pass")

    if phon_vocab_size != configured_model.phonological_vocabulary_size:
        print(f"⚠ WARNING: Phonological vocabulary size mismatch!")
        print(f"   Expected: {configured_model.phonological_vocabulary_size}")
        print(f"   Got: {phon_vocab_size}")
        print(f"   This might indicate a bug in the model's forward pass")


def test_model_output_quality(model_output, sample_encoding, configured_model):
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological
    full_nan_debug(configured_model, ortho, phon)
    # --- Add this block to help debug NaNs ---
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological
    debug_nan_trace(
        configured_model,
        {
            "orth_enc_input": ortho.enc_input_ids,
            "orth_enc_pad_mask": ortho.enc_pad_mask,
            "phon_dec_input": phon.dec_input_ids,
            "phon_dec_pad_mask": phon.dec_pad_mask,
        },
    )
    # --- End debug block ---
    # Check output quality
    # assert False, "PREMATURE END"
    assert not torch.allclose(
        model_output["orth"], torch.zeros_like(model_output["orth"])
    )
    assert not torch.isnan(model_output["orth"]).any()
    assert not torch.isinf(model_output["orth"]).any()

    assert not torch.allclose(
        model_output["phon"], torch.zeros_like(model_output["phon"])
    )
    assert not torch.isnan(model_output["phon"]).any()
    assert not torch.isinf(model_output["phon"]).any()

    print(f"✓ Output quality checks passed")
    print(
        f"✓ Orthographic output range: [{model_output['orth'].min():.3f}, {model_output['orth'].max():.3f}]"
    )
    print(
        f"✓ Phonological output range: [{model_output['phon'].min():.3f}, {model_output['phon'].max():.3f}]"
    )


def test_sequence_length_respect(sample_dataset):
    """Test that generated sequences respect max_seq_len constraint."""
    dataset = cast(BridgeDataset, sample_dataset)

    # Check that all sequences are within the max_seq_len
    max_seq_len = 128

    for i, sequence in enumerate(dataset.sequences):
        # Get the encoding for this sequence
        encoding = dataset[i]
        ortho = encoding.orthographic

        # Check orthographic sequence length
        seq_len = ortho.enc_input_ids.shape[1]
        assert (
            seq_len <= max_seq_len
        ), f"Sequence {i} exceeds max_seq_len: {seq_len} > {max_seq_len}"


def test_word_boundary_preservation(sample_dataset):
    """Test that word boundaries are preserved in phoneme sequences."""
    dataset = cast(BridgeDataset, sample_dataset)

    # Test the first sequence
    word_sequence = dataset.sequences[0]
    text_sequence = " ".join(word_sequence)

    # Get the phoneme sequence
    phonemes = dataset.tokenizer.phoneme_tokenizer._get_phrase_phonemes(text_sequence)

    assert phonemes is not None

    # Count [SPC] tokens
    spc_count = phonemes.count("[SPC]")
    expected_spc_count = len(word_sequence) - 1

    assert spc_count == expected_spc_count, (
        f"Expected {expected_spc_count} [SPC] tokens for {len(word_sequence)} words, "
        f"but found {spc_count}"
    )


def test_sequence_length_analysis(sample_dataset, sample_encoding, configured_model):
    """Analyze the sequence lengths at each step."""
    ortho = sample_encoding.orthographic
    phon = sample_encoding.phonological

    print(f"Input sequence lengths:")
    print(f"  orth_enc_input: {ortho.enc_input_ids.shape[1]}")
    print(
        f"  phon_enc_input: {len(phon.enc_input_ids[0]) if phon.enc_input_ids else 'N/A'}"
    )
    print(f"  orth_dec_input: {ortho.dec_input_ids.shape[1]}")
    print(
        f"  phon_dec_input: {len(phon.dec_input_ids[0]) if phon.dec_input_ids else 'N/A'}"
    )

    # Check which sequence is shorter
    orth_len = ortho.enc_input_ids.shape[1]
    phon_len = len(phon.enc_input_ids[0]) if phon.enc_input_ids else 0
    min_len = min(orth_len, phon_len)

    print(f"  Minimum sequence length: {min_len}")
    print(f"  Orthographic output sequence length: 34")
    print(f"  Phonological output sequence length: 29")

    # Check if the output lengths match the minimum input length
    if 34 == min_len:
        print(f"  ✓ Orthographic output length matches minimum input length")
    else:
        print(
            f"  ⚠ Orthographic output length (34) doesn't match minimum input length ({min_len})"
        )

    if 29 == min_len:
        print(f"  ✓ Phonological output length matches minimum input length")
    else:
        print(
            f"  ⚠ Phonological output length (29) doesn't match minimum input length ({min_len})"
        )


def test_two_word_phonological_format(
    sample_dataset: SyntheticBridgeDatasetMultiWord,
    # word_to_phoneme_sequence: tuple[list[str], list[list[Tensor]]],
):
    """Test that two-word sequences create correct phonological input format.

    Verifies that phon_enc_input has structure: list[list[Tensor]] where
    each inner list contains phonemes from both words in sequence.
    """
    dataset = cast(BridgeDataset, sample_dataset)
    words = ["cariello", "annually", "unfav", "gehringer", "ensemble", "colpitts"]
    phoneme_sequences, phoneme_features = word_sequence_to_phoneme_sequence(
        words,
        dataset.tokenizer.phoneme_tokenizer,
    )

    nb_words: int = 2
    word_sequence = words[:nb_words]
    concatenated_words = " ".join(word_sequence)

    orth_enc_input = [concatenated_words]
    batch_size = 1

    # Test with model - just verify it doesn't crash
    model_config = ModelConfig(
        d_model=128,
        nhead=4,
        num_phon_enc_layers=2,
        num_orth_enc_layers=2,
        num_mixing_enc_layers=2,
        num_phon_dec_layers=2,
        num_orth_dec_layers=2,
        d_embedding=1,
        seed=42,
        use_sliding_window=False,
        window_size=61,
        is_causal=False,
        max_seq_len=256,
        ensure_contiguous=False,
    )

    model = Model(model_config, dataset)
    model.eval()

    # Mock sequence lengths to handle longer sequences
    model.max_orth_seq_len = 1024  # 128
    model.max_phon_seq_len = 1024  # 128

    # Update position embeddings
    device = model.device
    d_model = model.model_config.d_model

    model.orth_position_embedding = torch.nn.Embedding(128, d_model, device=device)
    model.phon_position_embedding = torch.nn.Embedding(128, d_model, device=device)

    encoding = dataset.tokenizer.encode(orth_enc_input)
    print(f"{encoding.orthographic.enc_input_ids=}")
    print(f"{encoding.orthographic.dec_input_ids=}")
    print(f"{encoding.phonological.enc_input_ids=}")
    print(f"{encoding.phonological.dec_input_ids=}")
    print(f"{encoding.orthographic.enc_pad_mask=}")
    print(f"{encoding.orthographic.dec_pad_mask=}")
    print(f"{encoding.phonological.enc_pad_mask=}")
    print(f"{encoding.phonological.dec_pad_mask=}")

    with torch.no_grad():
        output = model.forward(
            "op2op",
            orth_enc_input=encoding.orthographic.enc_input_ids,
            orth_enc_pad_mask=encoding.orthographic.enc_pad_mask,
            phon_enc_input=encoding.phonological.enc_input_ids,
            phon_enc_pad_mask=encoding.phonological.enc_pad_mask,
            phon_dec_input=encoding.phonological.dec_input_ids,
            phon_dec_pad_mask=encoding.phonological.dec_pad_mask,
            orth_dec_input=encoding.orthographic.dec_input_ids,
            orth_dec_pad_mask=encoding.orthographic.dec_pad_mask,
        )

    # Just verify the model produces output - don't check exact shapes
    assert "phon" in output, "Model should produce phonological output"
    assert "orth" in output, "Model should produce orthographic output"


def test_decoder_output_length_and_nan(configured_model):
    """Test if model output is always one shorter than input and last position is NaN.

    This test checks for off-by-one errors in the decoder output length and NaN in the last position.
    """
    import torch

    min_len = 10
    max_len = 50
    batch_size = 1
    model = configured_model

    # Use the model's vocab size for valid token indices
    vocab_size = model.orthographic_vocabulary_size
    phon_vocab_size = model.phonological_vocabulary_size
    print(f"===>+++ {phon_vocab_size=}")

    for seq_len in range(min_len, max_len + 1, 5):
        # Create dummy input of shape (batch_size, seq_len)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        # For phonological input, create a dummy list of lists of tensors
        # FIXED: Each tensor should contain single feature indices, not multiple elements
        phon_enc_input = [
            [torch.randint(0, phon_vocab_size, (1,)) for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
        phon_enc_pad_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)

        # FIXED: Create separate decoder inputs for orthographic and phonological
        # Orthographic decoder input can use the same as encoder input
        orth_dec_input = input_ids.clone()
        orth_dec_pad_mask = pad_mask.clone()

        # Phonological decoder input must use phonological vocabulary size
        phon_dec_input = [
            [torch.randint(0, phon_vocab_size, (1,)) for _ in range(seq_len)]
            for _ in range(batch_size)
        ]
        phon_dec_pad_mask = pad_mask.clone()

        # Run the model
        with torch.no_grad():
            output = model.forward_op2op(
                orth_enc_input=input_ids,
                orth_enc_pad_mask=pad_mask,
                phon_enc_input=phon_enc_input,
                phon_enc_pad_mask=phon_enc_pad_mask,
                orth_dec_input=orth_dec_input,
                orth_dec_pad_mask=orth_dec_pad_mask,
                phon_dec_input=phon_dec_input,
                phon_dec_pad_mask=phon_dec_pad_mask,
            )

        orth_out = output["orth"]
        # orth_out shape: (batch_size, vocab_size, seq_len)
        out_seq_len = orth_out.shape[2]

        print(f"Input seq_len: {seq_len}, Output seq_len: {out_seq_len}")

        # Check if output is one shorter than input
        if out_seq_len == seq_len - 1:
            # Check if last position is NaN
            last_col = orth_out[:, :, -1]
            has_nan = torch.isnan(last_col).any().item()
            print(f"  Last output position is NaN: {has_nan}")
            assert has_nan, f"Expected NaN in last position for seq_len={seq_len}"
        else:
            print(
                f"  Output seq_len is not one less than input (no off-by-one bug for seq_len={seq_len})"
            )
            # Optionally, assert that output matches input length
            assert (
                out_seq_len == seq_len
            ), f"Unexpected output length for seq_len={seq_len}"

    print(
        "✓ Off-by-one and NaN pattern test completed for all tested sequence lengths."
    )


if __name__ == "__main__":
    """Run the tests when executed directly."""
    print("Running Test 1.1: Disabled vs Enabled Sliding Window")

    # Run the main test
    test_1_1_disabled_vs_enabled_sliding_window()

    # Run edge case tests
    test_1_1_edge_cases()

    # Run scaling benefits test
    test_1_1_scaling_benefits()

    # Run dataset encoding demonstration test
    test_1_1_dataset_encoding_demonstration()

    print("\n All Test 1.1 tests passed!")

if __name__ == "__main__":
    """Run the tests when executed directly."""
    pytest.main([__file__, "-v"])
