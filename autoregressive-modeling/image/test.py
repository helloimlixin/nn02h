# testing pipeline
from models.pixelcnn import PixelCNN
import torch
from torchvision.utils import save_image, make_grid
import config

if __name__ == "__main__":
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = PixelCNN.load_from_checkpoint("checkpoints/pixelcnn/last.ckpt")

    # sample from the model
    label_indices = torch.randint(0, 10, (4,), device=model.device)
    sample = model.generate(label_indices, (4, 3, 32, 32))
    print(f"Generated samples for classes: {', '.join([classes[i] for i in label_indices.cpu().numpy()])}")

    # save the sample
    save_image(make_grid(sample, nrow=4), "sample.png")