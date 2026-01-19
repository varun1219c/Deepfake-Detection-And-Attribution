import torch
import sys

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN Version: {torch.backends.cudnn.version()}")

if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Test GPU computation
    x = torch.rand(5, 3).cuda()
    print(f"\n✓ GPU Test Successful!")
    print(f"Test Tensor on GPU:\n{x}")
else:
    print("\n✗ GPU not detected!")