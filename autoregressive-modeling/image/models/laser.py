import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.terminal.shortcuts.filters import eval_node


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Compute distances and find nearest embedding
        distances = (
            torch.sum(flat_input ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(flat_input, self.embeddings.weight.t())
        )
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        # encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        # encodings.scatter_(1, encoding_indices, 1)

        # # Quantize
        # quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        #
        # # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        #
        # quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encoding_indices
        return encoding_indices

    def quantize(self, encoding_indices, latent_dim=(8, 8)):
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        # quantized = torch.matmul(encodings, self.embeddings.weight).view(input_shape)
        quantized = torch.matmul(encodings, self.embeddings.weight)

        # Loss
        # e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        # q_latent_loss = F.mse_loss(quantized, inputs.detach())
        # loss = q_latent_loss + self._commitment_cost * e_latent_loss
        #
        # quantized = inputs + (quantized - inputs).detach()
        # avg_probs = torch.mean(encodings, dim=0)
        # perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, encodings

    def loss(self, quantized, encodings, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    pass

class GumbelQuantizer(nn.Module):
    pass

