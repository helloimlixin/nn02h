import torch
import torch.nn as nn
import torch.nn.functional as F


class OnlineDictionaryLearning(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, sparsity, ema_decay=0.99, tolerance=1e-7, lr=0.01, debug=False):
        """
        Online Dictionary Learning with Batch OMP and Exponential Moving Average (EMA) for cluster size.

        Args:
            embedding_dim (int): Dimensionality of input signals.
            num_embeddings (int): Number of dictionary atoms.
            sparsity (int): Target sparsity level.
            ema_decay (float): Exponential moving average decay for cluster size.
            tolerance (float): Convergence tolerance for Batch OMP.
            lr (float): Learning rate for dictionary updates.
            debug (bool): Enable debugging information.
        """
        super(OnlineDictionaryLearning, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self._sparsity_level = sparsity
        self.ema_decay = ema_decay
        self.tolerance = tolerance
        self.lr = lr
        self.debug = debug

        # Initialize dictionary and cluster size
        dictionary = torch.randn(embedding_dim, num_embeddings)
        self.dictionary = nn.Parameter(dictionary, requires_grad=False)
        self.cluster_size = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.dictionary_avg = nn.Parameter(dictionary.clone(), requires_grad=False)
        self._normalize_dictionary()

    def _normalize_dictionary(self):
        """Normalize dictionary columns to unit norm."""
        with torch.no_grad():
            self.dictionary.data = F.normalize(self.dictionary.data, dim=0)

    def batch_omp(self, X):
        """
        Perform Batch OMP to compute sparse codes.

        Args:
            X (torch.Tensor): Input signals of shape (embedding_dim, num_signals).

        Returns:
            torch.Tensor: Sparse codes of shape (num_signals, num_embeddings).
        """
        embedding_dim, num_signals = X.size()
        dictionary_t = self.dictionary.T
        gram_matrix = dictionary_t @ self.dictionary
        corr_init = (dictionary_t @ X).T  # Initial correlation

        corr = corr_init

        sparse_codes = torch.zeros_like(corr_init)
        L = torch.ones(num_signals, 1, 1,
                       device=X.device)  # contains the progressive Cholesky of the Gram matrix in the selected indices
        I = torch.zeros(num_signals, 0, dtype=torch.long, device=X.device)  # placeholder for the index set
        omega = torch.ones_like(corr_init, dtype=torch.bool)  # used to zero out elements in corr before argmax
        signal_idx = torch.arange(num_signals, device=X.device)
        delta = torch.zeros(num_signals, device=X.device)
        eps = torch.norm(X, dim=0)  # the residual, initialized as the L2 norm of the signal

        k = 0
        while k < self._sparsity_level:
            k += 1
            k_hats = torch.argmax(torch.abs(corr * omega), dim=1)  # select the index of the maximum correlation
            # update omega to make sure we do not select the same index twice
            omega[torch.arange(k_hats.shape[0], device=X.device), k_hats] = 0
            expanded_signal_idx = signal_idx.unsqueeze(0).expand(k,
                                                                 num_signals).t()  # expand is more efficient than repeat

            if k > 1:  # Cholesky update
                G_ = gram_matrix[I[signal_idx, :], k_hats[expanded_signal_idx[..., :-1]]].view(num_signals, k - 1,
                                                                                               1)  # compute for all signals in a vectorized manner
                w = torch.linalg.solve_triangular(L, G_, upper=False).view(-1, 1, k - 1)
                w_br = torch.sqrt(
                    1 - (w ** 2).sum(dim=2, keepdim=True))  # L bottom-right corner element: sqrt(1 - w.t().mm(w))

                # concatenate into the new Cholesky: L <- [[L, 0], [w, w_br]]
                k_zeros = torch.zeros(num_signals, k - 1, 1, device=X.device)
                L = torch.cat((
                    torch.cat((L, k_zeros), dim=2),
                    torch.cat((w, w_br), dim=2),
                ), dim=1)

            # update non-zero indices
            I = torch.cat([I, k_hats.unsqueeze(1)], dim=1)

            # solve L
            corr_ = corr_init[expanded_signal_idx, I[signal_idx, :]].view(num_signals, k, 1)
            gamma_ = torch.cholesky_solve(corr_, L)

            # de-stack sparse_codes into the non-zero elements
            sparse_codes[signal_idx.unsqueeze(1), I[signal_idx]] = gamma_[signal_idx].squeeze(-1)

            # beta = G_I * gamma_I
            beta = sparse_codes[signal_idx.unsqueeze(1), I[signal_idx]].unsqueeze(1).bmm(
                gram_matrix[I[signal_idx], :]).squeeze(1)

            corr = corr_init - beta

            # update residual
            new_delta = (sparse_codes * beta).sum(dim=1)
            eps += delta - new_delta
            delta = new_delta

            if self.debug and k % 1 == 0:
                print('Step {}, residual: {:.4f}, below tolerance: {:.4f}'.format(k, eps.max(),
                                                                                  (eps < 1e-7).float().mean().item()))

        return sparse_codes.t()  # transpose the sparse coefficients to make num_signals the first dimension

    def update_dictionary(self, X, Z):
        """
        Update the dictionary using Exponential Moving Average (EMA).

        Args:
            X (torch.Tensor): Input signals of shape (batch_size, embedding_dim).
            Z (torch.Tensor): Sparse codes of shape (batch_size, num_embeddings).
        """
        self.cluster_size.data.mul_(self.ema_decay).add_(
            (1 - self.ema_decay) * Z.abs().sum(1)
        )

        embed_sum = X @ Z.T
        self.dictionary_avg.data.mul_(self.ema_decay).add_((1 - self.ema_decay) * embed_sum)

        n = self.cluster_size.sum()
        cluster_size = (
                (self.cluster_size + 1e-6) / (n + self.num_embeddings * 1e-6) * n
        )

        embed_normalized = self.dictionary_avg / cluster_size.unsqueeze(0)

        self.dictionary.data.copy_(embed_normalized)

        # Normalize dictionary columns
        self._normalize_dictionary()

        if self.debug:
            print(f"Cluster Size: {self.cluster_size}")
            print(f"Dictionary Avg Norm: {self.dictionary_avg.norm().item()}")

    def forward(self, X):
        """
        Forward pass: Sparse coding and dictionary update.

        Args:
            X (torch.Tensor): Input signals of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Sparse codes of shape (batch_size, num_embeddings).
        """
        Z = self.batch_omp(X)
        self.update_dictionary(X, Z)
        return Z
