import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, epsilon=1e-7):
        super().__init__()
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._commitment_cost = commitment_cost

        self._embeddings = nn.Embedding(num_embeddings, embedding_dim)  # codebook of dimension K x D
        self._embeddings.weight.data.uniform_(-1./num_embeddings, 1./num_embeddings)
        self._epsilon = epsilon

    def code(self, inputs):
        flat_input = inputs.view(-1, self._embedding_dim)  # (B x H x W) x D

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) +
                     torch.sum(self._embeddings.weight**2, dim=1) -
                     2 * torch.matmul(flat_input, self._embeddings.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        return encoding_indices

    def quantize(self, encodings):
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embeddings.weight)

        return quantized

    def forward(self, inputs):
        # inputs: B x D x H x W
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        # Encoding
        encodings = self.code(inputs)

        # Quantize and unflatten
        quantized = self.quantize(encodings).view(inputs.shape)  # B x D x H x W

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + self._epsilon)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class VectorQuantizerEMA(nn.Module):
    pass

class GumbelQuantizer(nn.Module):
    pass

