import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Number of CUDA devices:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))

    # Create a tensor on the GPU
    x = torch.tensor([1, 2, 3], device="cuda")
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available.")
