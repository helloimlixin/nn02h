import lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from .utils import WaveNetModule

class WaveNet(pl.LightningModule):
    def __init__(self, in_channels, residual_channels, skip_channels, num_blocks):
        super(WaveNet, self).__init__()

        self.net = WaveNetModule(in_channels, residual_channels, skip_channels, num_blocks)

    def _inference_step(self, batch):
        batch = batch.unsqueeze(1).to(torch.float32)
        outputs = self.net(batch)
        targets = batch[:, :, 1:]
        loss = F.cross_entropy(outputs[:, :, :-1], targets)
        return loss

    @torch.no_grad()
    def generate_audio(self, seed, num_samples):
        self.eval()
        generated = seed  # Initial waveform seed

        for _ in range(num_samples):
            with torch.no_grad():
                output = self.forward(generated.unsqueeze(0).unsqueeze(0))  # Add batch and channel dims
                next_sample = output[:, :, -1].argmax(dim=1)  # Get most likely next value
                generated = torch.cat([generated, next_sample], dim=-1)  # Append prediction

        return generated

    def training_step(self, batch, batch_idx):
        loss = self._inference_step(batch)

        self.log_dict({'train_loss': loss},
                        on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._inference_step(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._inference_step(batch)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
