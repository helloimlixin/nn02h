import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dl_ema import OnlineDictionaryLearning
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Set up training configuration
batch_size = 32
embedding_dim = 784
num_embeddings = 256
sparsity = 5  # Target sparsity level for sparse codes
num_epochs = 1000  # Number of training epochs
learning_rate = 0.1  # Learning rate for dictionary updates

# Create synthetic dataset (random input signals)
X_train = torch.randn(1000, embedding_dim)  # 1000 signals, each of size signal_size

# Split the dataset into training and testing sets
train_dataset = TensorDataset(X_train[:800])
test_dataset = TensorDataset(X_train[800:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the Online Dictionary Learning model
model = OnlineDictionaryLearning(
    embedding_dim=embedding_dim,
    num_embeddings=num_embeddings,
    sparsity=sparsity,
    ema_decay=0.99,
    tolerance=1e-7,
    lr=learning_rate,
    debug=False
)

# Set up the optimizer (optional, if you want to use it for other parts of the model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_reconstruction_error = 0
    total_sparse_codes_norm = 0
    total_dictionary_norm = 0

    for i, (X_batch,) in enumerate(train_loader):
        # Zero the gradients (if using an optimizer)
        optimizer.zero_grad()

        # Forward pass (get sparse codes and dictionary updates)
        sparse_codes = model(X_batch.T)

        # Compute reconstruction error (optional for monitoring progress)
        reconstruction_error = torch.norm(X_batch - sparse_codes.T @ model.dictionary.T, p='fro')

        # Track the norms for monitoring
        sparse_codes_norm = sparse_codes.norm(p='fro')
        dictionary_norm = model.dictionary.norm(p='fro')

        # Accumulate loss values for averaging later
        total_reconstruction_error += reconstruction_error.item()
        total_sparse_codes_norm += sparse_codes_norm.item()
        total_dictionary_norm += dictionary_norm.item()

        # Backpropagate (if necessary)
        # reconstruction_error.backward()  # For dictionary learning tasks, this may not be needed directly

    # Print out metrics for this epoch
    avg_reconstruction_error = total_reconstruction_error / len(train_loader)
    avg_sparse_codes_norm = total_sparse_codes_norm / len(train_loader)
    avg_dictionary_norm = total_dictionary_norm / len(train_loader)

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Avg Reconstruction Error: {avg_reconstruction_error:.4f}")
    print(f"  Avg Sparse Codes Norm: {avg_sparse_codes_norm:.4f}")
    print(f"  Avg Dictionary Norm: {avg_dictionary_norm:.4f}")

    # Optional: Save model checkpoint
    # torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pth')

print("Training complete!")

# Evaluate the model on the test set
model.eval()
total_test_reconstruction_error = 0
total_test_sparse_codes_norm = 0
total_test_dictionary_norm = 0

for i, (X_batch,) in enumerate(test_loader):
    # Forward pass (get sparse codes and dictionary updates)
    sparse_codes = model(X_batch.T)

    # Compute reconstruction error (optional for monitoring progress)
    reconstruction_error = torch.norm(X_batch - sparse_codes.T @ model.dictionary.T, p='fro')

    # Track the norms for monitoring
    sparse_codes_norm = sparse_codes.norm(p='fro')
    dictionary_norm = model.dictionary.norm(p='fro')

    # Accumulate loss values for averaging later
    total_test_reconstruction_error += reconstruction_error.item()
    total_test_sparse_codes_norm += sparse_codes_norm.item()
    total_test_dictionary_norm += dictionary_norm.item()

# Print out metrics for the test set
avg_test_reconstruction_error = total_test_reconstruction_error / len(test_loader)
avg_test_sparse_codes_norm = total_test_sparse_codes_norm / len(test_loader)
avg_test_dictionary_norm = total_test_dictionary_norm / len(test_loader)

print("Evaluation on test set:")
print(f"  Avg Reconstruction Error: {avg_test_reconstruction_error:.4f}")
print(f"  Avg Sparse Codes Norm: {avg_test_sparse_codes_norm:.4f}")
print(f"  Avg Dictionary Norm: {avg_test_dictionary_norm:.4f}")
