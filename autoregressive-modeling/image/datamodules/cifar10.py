from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import lightning as pl
from torchvision.transforms import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image, make_grid


def discretize(x):
    return (x * 255).clamp(0, 255).long()


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        entire_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    discretize
                ]
            ),
            download=False,
        )

        # save first 8 images for visualization
        images = [entire_dataset[i][0] / 255.0 for i in range(8)]
        grid = make_grid(images, nrow=4)
        save_image(grid, "cifar10.png")

        self.train_dataset, self.val_dataset = random_split(
            entire_dataset, [45_000, 5_000]
        )

        self.test_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    discretize
                ]
            ),
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
