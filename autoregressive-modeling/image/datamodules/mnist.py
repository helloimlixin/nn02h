import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import lightning as pl


def discretize(images):
    return images * 255  # [0, 1] -> [0, 255]

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # multi gpus, load after prepare_data
        entire_dataset = datasets.MNIST(
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

        self.train_dataset, self.val_dataset = random_split(
            entire_dataset, [55_000, 5_000]
        )

        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers  # uncomment this line if you are using cluster
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size // 8,
            shuffle=False,
            num_workers=self.num_workers,  # uncomment this line if you are using cluster
            # persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size // 8,
            shuffle=False,
            num_workers=self.num_workers  # uncomment this line if you are using cluster
        )
