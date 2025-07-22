import torch


def check_nan(tensor: torch.Tensor | None, name: str) -> None:
    if tensor is None:
        print(f"✅ tensor provided is None for {name}!")
        return
    if torch.isnan(tensor).any():
        print(f"Check_nan: {name}")
        print("    ⚠ WARNING: NaN detected!")
        nan_count = torch.isnan(tensor).sum().item()
        print(f"    ⚠ NaN count: {nan_count}/{tensor.numel()}")
        print(f"    ⚠ Tensor shape: {tensor.shape}")
    else:
        print(f"✅ No NaN detected in {name}, {tensor.shape}!")
        return
