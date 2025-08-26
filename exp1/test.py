import torch, sys
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("is_available:", torch.cuda.is_available())
print("python:", sys.executable)