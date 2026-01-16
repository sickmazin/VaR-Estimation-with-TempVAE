import torch
from TempVae import TempVAE

model = TempVAE(
    input_dim=4,
    latent_dim=10,
    hidden_dim=16,
    dropout=0.1
)

print("="*50)
print("TEMP VAE ARCHITECTURE")
print("="*50)
print(model)

print("\n" + "="*50)
print("PARAMETER COUNT")
print("="*50)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total Parameters:     {total_params}")
print(f"Trainable Parameters: {trainable_params}")
print(f"Fixed (Prior) Params: {total_params - trainable_params}")
print("="*50)