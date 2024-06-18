from torch.cuda import is_available

print("Cuda is available: " + ("YES" if is_available() else "NO"))
