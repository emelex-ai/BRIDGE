import torch
from torch import nn


def create_sliding_window_causal_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Create a sliding window causal mask with 0.0/-inf values, shape (seq_len, seq_len)."""
    positions = torch.arange(seq_len, device=device)
    query_pos = positions.unsqueeze(1)
    key_pos = positions.unsqueeze(0)
    causal_mask = key_pos <= query_pos
    window_mask = (key_pos >= query_pos - window_size + 1) & (key_pos <= query_pos)
    combined_mask = causal_mask & window_mask
    return torch.where(combined_mask, 0.0, float("-inf")).to(torch.float32)


def test_decoder_layer_nan_batch_first_bool_mask():
    """Test a single TransformerDecoderLayer with batch_first=True and sliding window mask for NaN propagation."""
    d_model = 128
    nhead = 4
    seq_len = 1024
    batch_size = 2
    window_size = 61
    device = torch.device("cpu")

    # Create random input (mimic model output)
    tgt = torch.randn(batch_size, seq_len, d_model, device=device)
    memory = torch.randn(batch_size, seq_len, d_model, device=device)

    # Create float mask and convert to bool
    mask = create_sliding_window_causal_mask(seq_len, window_size, device)
    mask_bool = mask == 0.0  # True where attend, False where mask

    # Instantiate a single decoder layer with batch_first=True
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=d_model, nhead=nhead, batch_first=True
    )
    decoder_layer.eval()

    # Run the decoder layer
    with torch.no_grad():
        out = decoder_layer(tgt, memory, tgt_mask=mask_bool)
        print(f"Output shape: {out.shape}")
        print(f"NaN in output: {torch.isnan(out).any()}")
        if torch.isnan(out).any():
            nan_count = torch.isnan(out).sum().item()
            total = out.numel()
            print(f"NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")
        assert not torch.isnan(
            out
        ).any(), f"NaN detected in decoder output: {nan_count}/{total} elements"


if __name__ == "__main__":
    test_decoder_layer_nan_batch_first_bool_mask()
