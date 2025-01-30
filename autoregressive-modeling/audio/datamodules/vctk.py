import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torchaudio.functional as F

class VCTKDataset(Dataset):
    def __init__(self, root_dir, target_sample_rate=16000, segment_length=16000, quantization_channels=256):
        """
        Args:
            root_dir: Path to VCTK dataset (contains the wav files).
            target_sample_rate: Audio sample rate to resample to (default is 16kHz).
            segment_length: Length of each audio segment in samples (1 second = 16,000).
            quantization_channels: Number of channels for mu-law encoding (default is 256).
        """
        self.root_dir = root_dir
        self.sample_rate = target_sample_rate
        self.segment_length = segment_length
        self.quantization_channels = quantization_channels

        # Collect all .wav file paths
        self.audio_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".wav"):
                    self.audio_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load and resample audio
        audio_path = self.audio_files[idx]
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to target sample rate if necessary
        if sample_rate != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)

        # Randomly select a segment of fixed length
        if waveform.size(1) > self.segment_length:
            max_start = waveform.size(1) - self.segment_length
            start = torch.randint(0, max_start, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length]
        else:
            # Pad if the segment is too short
            padding = self.segment_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding))

        # Mu-law encoding
        mu_law_encoded = F.mu_law_encoding(waveform.squeeze(0), self.quantization_channels)

        return mu_law_encoded


import torch
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class VCTKDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size=16, num_workers=4, target_sample_rate=16000, 
                 segment_length=16000, quantization_channels=256, train_val_split=0.8):
        """
        Args:
            root_dir: Path to the VCTK dataset.
            batch_size: Batch size for data loaders.
            num_workers: Number of workers for data loading.
            target_sample_rate: Target sample rate to resample audio.
            segment_length: Length of audio segments.
            quantization_channels: Number of channels for mu-law encoding.
            train_val_split: Proportion of the dataset used for training (default is 80% train).
        """
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_sample_rate = target_sample_rate
        self.segment_length = segment_length
        self.quantization_channels = quantization_channels
        self.train_val_split = train_val_split

    def setup(self, stage=None):
        # Instantiate the dataset
        full_dataset = VCTKDataset(
            root_dir=self.root_dir,
            target_sample_rate=self.target_sample_rate,
            segment_length=self.segment_length,
            quantization_channels=self.quantization_channels
        )

        # Split the dataset into training and validation sets
        train_size = int(self.train_val_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def test_dataloader(self):
        # Assuming test data is stored separately, handle accordingly (optional)
        return None  # Replace if you have a test dataset


if __name__ == "__main__":
    # Initialize the VCTK data module
    dm = VCTKDataModule(
        root_dir="../../data/VCTK/VCTK-Corpus/VCTK-Corpus",
        batch_size=4,
        num_workers=0
    )

    # Setup the data module
    dm.setup()

    # Fetch a batch of data
    train_loader = dm.train_dataloader()
    for x in train_loader:
        print(x.size())
        break
