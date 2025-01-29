import torch
from torchvision.utils import save_image, make_grid
from models.pixelcnn import PixelCNN
from models.laser_vae import VQVAE
import matplotlib.pyplot as plt
from datamodules.cifar10 import CIFAR10DataModule

dm = CIFAR10DataModule("data", batch_size=1, num_workers=0)
dm.prepare_data()
dm.setup()
val_loader = dm.val_dataloader()

test_img = torch.zeros((3, 32, 32)).cuda()
# test_img[:, 16, 16] = 1  # Set one pixel to max value

# model = PixelCNN.load_from_checkpoint("checkpoints/pixelcnn/last.ckpt").cuda()
#
# print(model.eval())
#
# # generated_img = model.generate(labels=torch.tensor((1,), device="cuda").long(), img_size=test_img.shape, img=test_img.long())
# # save_image(generated_img / 255.0, "generated_image.png")
#
# logits = model(test_img.long(), labels=torch.tensor((1,), device="cuda").long())
# probs = torch.softmax(logits, dim=-1)
# entropy = -torch.sum(probs * torch.log(probs), dim=-1).mean()
#
# print(entropy.item())

vae = VQVAE.load_from_checkpoint("checkpoints/vae/last.ckpt").cuda()
vae.eval()
sample_images, _ = next(iter(val_loader))  # Get a batch of CIFAR-10 images
sample_images = sample_images.cuda()
encoding_indices, z = vae.encode(sample_images)

sample_images = (sample_images + 1) / 2
save_image(make_grid(sample_images, nrow=8), "original_images.png")

quantized, encodings = vae.decode(encoding_indices)
quantized = quantized.view(-1, 8, 8, 64).permute(0, 3, 1, 2).contiguous()
# loss, quantized, perplexity, encodings = vae.quantizer.loss(quantized, encodings, z)
decoded_images1 = vae.decoder(quantized)
decoded_images1 = decoded_images1.detach().cpu()
decoded_images1 = (decoded_images1 + 1) / 2
decoded_images2 = vae(sample_images)[1].detach().cpu()
save_image(make_grid(decoded_images1, nrow=8), "reconstructed_images.png")

