"""Rode no terminal do VSCode e tire o print junto com nvidia-smi e a extensão GPU Monitor."""
import torch
print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
print("CUDA do PyTorch:", torch.version.cuda)
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print(f"VRAM total: {p.total_memory/1024**3:.1f} GB")
    x = torch.randn(4096, 4096, device="cuda")
    print("Teste de matmul na GPU OK:", (x @ x).shape)
else:
    print("ATENÇÃO: GPU não detectada. Veja docs/SETUP_GPU.md")
