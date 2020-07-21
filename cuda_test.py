import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Предполгаем, что у нас CUDA машина,
# поэтому должно напечататься CUDA устройство:

print(device)