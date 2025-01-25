import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineDictionaryLearning(nn.Module):
    def __init__(self, signal_size, dictionary_size, sparsity, ema_decay=0.99, tolerance=1e-7, lr=0.01, debug=False):
        """
        Online Dictionary Learning with Batch OMP and Exponential Moving Average (EMA) for cluster size.

        Args:
            signal_size (int): Dimensionality of input signals.
            dictionary_size (int): Number of dictionary atoms.
            sparsity (int): Target sparsity level.
            ema_decay (float): Exponential moving average decay for cluster size.
            tolerance (float): Convergence tolerance for Batch OMP.
            lr (float): Learning rate for dictionary updates.
            debug (bool): Enable debugging information.
        """
        super(OnlineDictionaryLearning, self).__init__()
        self.signal_size = signal_size
        self.dictionary_size = dictionary_size
        self.sparsity = sparsity
        self.ema_decay = ema_decay
        self.tolerance = tolerance
        self.lr = lr
        self.debug = debug

        # Initialize dictionary and cluster size
        self.dictionary = nn.Parameter(torch.randn(signal_size, dictionary_size))
        self.cluster_size = nn.Parameter(torch.zeros(dictionary_size), requires_grad=False)
        self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize dictionary columns to unit norm."""
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=0)

    def batch_omp(self, X):
        """
        Perform Batch OMP to compute sparse codes.

        Args:
            X (torch.Tensor): Input signals of shape (batch_size, signal_size).

        Returns:
            torch.Tensor: Sparse codes of shape (batch_size, dictionary_size).
        """
        batch_size, signal_size = X.size()
        dictionary_t = self.dictionary.T
        gram_matrix = dictionary_t @ self.dictionary
        gram_matrix += 1e-6 * torch.eye(gram_matrix.size(-1), device=gram_matrix.device)  # Regularization
        corr_init = (dictionary_t @ X.T).T  # Initial correlation

        sparse_codes = torch.zeros(batch_size, self.dictionary_size, device=X.device)

        try:
            # Main OMP loop
            corr = corr_init.clone()
            selected_indices = torch.zeros(batch_size, 0, dtype=torch.long, device=X.device)
            L = torch.ones(batch_size, 1, 1, device=X.device)
            omega = torch.ones_like(corr_init, dtype=torch.bool)  # Valid atom mask
            batch_indices = torch.arange(batch_size, device=X.device)

            for k in range(self.sparsity):
                selected_atoms = torch.argmax((corr.abs() * omega), dim=1)
                omega[batch_indices, selected_atoms] = 0

                if k > 0:
                    G_stack = gram_matrix[selected_indices[:, :k], selected_atoms.unsqueeze(1)].view(batch_size, k, 1)
                    w = torch.linalg.solve_triangular(L, G_stack, upper=False).view(batch_size, 1, k)
                    w_bottom_right = torch.sqrt(1 - (w ** 2).sum(dim=2, keepdim=True))
                    L = torch.cat([
                        torch.cat([L, torch.zeros(batch_size, k, 1, device=X.device)], dim=2),
                        torch.cat([w, w_bottom_right], dim=2)
                    ], dim=1)

                selected_indices = torch.cat([selected_indices, selected_atoms.unsqueeze(1)], dim=1)

                corr_subset = torch.gather(corr_init, 1, selected_indices)
                gamma = torch.cholesky_solve(corr_subset.unsqueeze(-1), L).squeeze(-1)
                sparse_codes[batch_indices.unsqueeze(1), selected_indices] = gamma

                active_atoms = gram_matrix[selected_indices, :]
                beta_active = torch.bmm(gamma.unsqueeze(1), active_atoms).squeeze(1)
                beta = torch.zeros_like(corr_init)
                beta.scatter_(1, selected_indices, beta_active)
                corr = corr_init - beta

        except RuntimeError as e:
            print("Error in batch OMP:", e)
            sparse_codes = torch.zeros_like(sparse_codes)

        if torch.isnan(sparse_codes).any() or torch.isinf(sparse_codes).any():
            print("NaN or Inf detected in sparse codes! Resetting...")
            sparse_codes = torch.zeros_like(sparse_codes)

        return sparse_codes

    def update_dictionary(self, X, Z):
        """
        Update the dictionary using Exponential Moving Average (EMA).

        Args:
            X (torch.Tensor): Input signals of shape (batch_size, signal_size).
            Z (torch.Tensor): Sparse codes of shape (batch_size, dictionary_size).
        """
        with torch.no_grad():
            if torch.isnan(Z).any() or torch.isinf(Z).any():
                print("NaN or Inf detected in sparse codes! Resetting...")
                Z = torch.zeros_like(Z)

            self.cluster_size.data.mul_(self.ema_decay).add_(
                (1 - self.ema_decay) * Z.abs().sum(0)
            )

            self.cluster_size.data.clamp_min_(1e-6)

            embed_sum = X.T @ Z
            if not hasattr(self, "dictionary_avg"):
                self.dictionary_avg = nn.Parameter(torch.zeros_like(self.dictionary.data))
            self.dictionary_avg.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * embed_sum)

            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + 1e-6) / (n + self.dictionary_size * 1e-6) * n
            )

            embed_normalized = self.dictionary_avg / cluster_size.unsqueeze(0)
            embed_normalized = torch.clamp(embed_normalized, -1e3, 1e3)

            if torch.isnan(embed_normalized).any() or torch.isinf(embed_normalized).any():
                print("NaN or Inf detected in dictionary update! Resetting dictionary...")
                self.dictionary.data = torch.randn_like(self.dictionary.data)
                self._normalize_dictionary()
            else:
                self.dictionary.data.copy_(embed_normalized)

            if self.debug:
                print(f"Cluster Size: {self.cluster_size}")
                print(f"Dictionary Avg Norm: {self.dictionary_avg.norm().item()}")

    def forward(self, X):
        """
        Forward pass: Sparse coding and dictionary update.

        Args:
            X (torch.Tensor): Input signals of shape (batch_size, signal_size).

        Returns:
            torch.Tensor: Sparse codes of shape (batch_size, dictionary_size).
        """
        X = (X - X.mean(dim=1, keepdim=True)) / (X.std(dim=1, keepdim=True) + 1e-6)
        Z = self.batch_omp(X)
        self.update_dictionary(X, Z)
        return Z
