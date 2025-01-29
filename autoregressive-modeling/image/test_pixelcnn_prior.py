# testing pipeline
from models.pixelcnn import PixelCNN
from models.laser_vae import VQVAE
import torch
from torchvision.utils import save_image, make_grid
import config
import numpy as np

classes = None
if config.DATASET_NAME == "mnist":
    classes = [str(i) for i in range(10)]
else:
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

model = PixelCNN.load_from_checkpoint("checkpoints/pixelcnn/last.ckpt", strict=False)
ae = VQVAE.load_from_checkpoint("checkpoints/vae/last.ckpt")

# sample from the model
num_samples = config.NUM_SAMPLES
label_indices = torch.randint(0, config.NUM_CLASSES, (config.NUM_SAMPLES,), device=model.device)
latents = model.generate(label_indices, (config.NUM_SAMPLES, config.IMAGE_SIZE, config.IMAGE_SIZE)).view(-1, 1).contiguous()
quantized, encodings = ae.decode(latents)
quantized = quantized.view(-1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
samples = ae.decoder(quantized).detach().cpu()
samples = (samples + 1) / 2
print(f"Generated samples for classes: {', '.join([classes[i] for i in label_indices.cpu().numpy()])}")

# save the sample at the first RGB channel
latents = latents.view(-1, config.IMAGE_SIZE, config.IMAGE_SIZE) / 255.0
save_image(make_grid(latents[:, None]), "latents.png")
save_image(make_grid(samples, nrow=int(np.sqrt(config.NUM_SAMPLES))), "samples.png")