from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import lightning as pl
from torchvision.transforms import transforms


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = './data', batch_size: int = 32):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor())
            self.val_dataset = CIFAR10(self.data_dir, train=False, transform=transforms.ToTensor())
        if stage == 'test' or stage is None:
            self.test_dataset = CIFAR10(self.data_dir, train=False, transform=transforms.ToTensor())

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)