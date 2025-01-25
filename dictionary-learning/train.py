from dl_ema import OnlineDictionaryLearning
import torch
import torch.nn.functional as F

signal_size = 50
dictionary_size = 100
sparsity = 5
ema_decay = 0.99

model = OnlineDictionaryLearning(signal_size, dictionary_size, sparsity, lr=0.01, debug=True)

num_batches = 100
batch_size = 65536

for batch_idx in range(num_batches):
    X = torch.randn(batch_size, signal_size)  # Simulated input signals
    Z = model(X)

    reconstructed_X = Z @ model.dictionary.T
    reconstruction_error = torch.norm(X - reconstructed_X, dim=1).mean().item()
    print(f"Batch {batch_idx + 1}/{num_batches}, Reconstruction Error: {reconstruction_error:.6f}")








