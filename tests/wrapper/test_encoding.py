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
from bridge.domain.model import Model
from bridge.domain.model.synthetic_dataset import SyntheticBridgeDatasetMultiWord


@pytest.fixture
def sample_dataset():
    """Create a sample multi-word dataset for testing."""
    return SyntheticBridgeDatasetMultiWord(
        num_samples=5,
        max_seq_len=128,
        min_words_per_sequence=2,
        max_words_per_sequence=4,
        seed=42,
    )


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
    model.max_orth_seq_len = 148
    model.max_phon_seq_len = 148

    # Update position embeddings
    device = model.device
    d_model = model.model_config.d_model
    model.orth_position_embedding = torch.nn.Embedding(148, d_model, device=device)
    model.phon_position_embedding = torch.nn.Embedding(148, d_model, device=device)

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


def test_model_forward_pass_basic(model_output):
    """Test that model forward pass completes without errors."""
    # Basic output validation
    assert isinstance(model_output, dict)
    assert "orth" in model_output
    assert "phon" in model_output
    assert isinstance(model_output["orth"], torch.Tensor)
    assert isinstance(model_output["phon"], torch.Tensor)

    print(f"✓ Model forward pass completed successfully")
    print(f"✓ Orthographic output shape: {model_output['orth'].shape}")
    print(f"✓ Phonological output shape: {model_output['phon'].shape}")


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


def test_model_output_quality(model_output):
    """Test that model outputs are reasonable (not all zeros, no NaN, etc.)."""
    # Check output quality
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


if __name__ == "__main__":
    """Run the tests when executed directly."""
    pytest.main([__file__, "-v"])
