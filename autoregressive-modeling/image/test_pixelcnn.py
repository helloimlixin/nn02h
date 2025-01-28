# testing pipeline
from models.pixelcnn import PixelCNN
import torch
from torchvision.utils import save_image, make_grid
import config
import numpy as np

if __name__ == "__main__":
    classes = None
    if config.DATASET_NAME == "mnist":
        classes = [str(i) for i in range(10)]
    else:
        classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = PixelCNN.load_from_checkpoint("checkpoints/pixelcnn/last.ckpt")

    # sample from the model
    num_samples = config.NUM_SAMPLES
    label_indices = torch.randint(0, config.NUM_CLASSES, (config.NUM_SAMPLES,), device=model.device)
    samples = model.generate(label_indices, (config.NUM_SAMPLES, config.IMAGE_SIZE, config.IMAGE_SIZE))
    print(f"Generated samples for classes: {', '.join([classes[i] for i in label_indices.cpu().numpy()])}")

    # save the sample at the first RGB channel
    save_image(make_grid(samples[:, None], nrow=int(np.sqrt(config.NUM_SAMPLES))), "sample.png")