import torch

def batch_omp(X):
    """
    Perform Batch Orthogonal Matching Pursuit (OMP).

    Args:
        X (torch.Tensor): Input signals of shape (batch_size, signal_size).

    Returns:
        torch.Tensor: Sparse codes of shape (batch_size, dictionary_size).
    """
    batch_size, signal_size = X.shape
    _, dictionary_size = self.dictionary.shape

    # Initialize sparse codes
    Z = torch.zeros(batch_size, dictionary_size, device=X.device)

    # Initialize residuals as the input signals
    residuals = X.clone()

    # Selected atoms for each signal
    selected_atoms = torch.zeros(batch_size, self.sparsity, dtype=torch.long, device=X.device)

    for k in range(self.sparsity):
        # Step 1: Compute correlations between residuals and dictionary atoms
        correlations = torch.matmul(residuals, self.dictionary)  # Shape: (batch_size, dictionary_size)

        # Step 2: Select the atom with the highest absolute correlation for each signal
        selected_idx = torch.argmax(torch.abs(correlations), dim=1)  # Shape: (batch_size,)
        selected_atoms[:, k] = selected_idx

        # Step 3: Gather selected dictionary atoms for each signal
        active_atoms = self.dictionary[:, selected_atoms[:, :k+1]]  # Shape: (signal_size, batch_size, k+1)

        # Solve least squares problem for all signals in the batch
        active_atoms_T = active_atoms.permute(1, 0, 2)  # Shape: (batch_size, signal_size, k+1)
        active_coeffs = torch.linalg.lstsq(active_atoms_T, X.unsqueeze(2)).solution  # Shape: (batch_size, k+1, 1)

        # Update sparse codes with the active coefficients
        Z.scatter_(1, selected_atoms[:, :k+1], active_coeffs.squeeze(2))

        # Step 4: Update residuals
        residuals = X - torch.matmul(Z, self.dictionary.T)

    return Z
