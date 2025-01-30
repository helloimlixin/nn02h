import torch
import pandas as pd
import lightning as pl
from torch.utils.data import DataLoader


class LiverpoolIonSwitchingDataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.Series, sequence_length = 10):
        super().__init__()
        self.df = df
        self.sequence_length = sequence_length

    def __len__(self):
        return self.df.shape[0] - self.sequence_length - 1

    def __getitem__(self, idx):
        signal = self.df.iloc[idx:idx + self.sequence_length, 1].values
        label = self.df.iloc[idx + self.sequence_length - 1, 2]
        return signal, label


class LiverpoolIonSwitchingDataModule(pl.LightningDataModule):
    def __init__(self, train_data, test_data,
                 batch_size: int = 32,
                 train_split: float = 0.8,
                 sequence_length: int = 10,
                 num_workers: int = 4):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.train_split = train_split
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        train_dataset = LiverpoolIonSwitchingDataset(self.train_data, sequence_length=self.sequence_length)
        test_dataset = LiverpoolIonSwitchingDataset(self.test_data, sequence_length=self.sequence_length)

        train_size = int(self.train_split * len(train_dataset))
        val_size = len(train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        self.test_dataset = test_dataset

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