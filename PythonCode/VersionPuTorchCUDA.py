import torch
import os

# Проверка версии PyTorch
print(f"PyTorch version: {torch.__version__}")

# Проверка, доступна ли CUDA
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    # Проверка версии CUDA
    cuda_version = torch.version.cuda
    print(f"CUDA version: {cuda_version}")

    # Проверка количества доступных GPU
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")

    # Проверка информации о каждом GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available.")