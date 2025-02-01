import time
import sys
import urllib.request
import tarfile
import os
import pandas as pd
from packaging import version
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import lightning as pl

start_time = time.time()

def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = progress_size / (1024.0**2 * duration)
    percent = count * block_size * 100.0 / total_size
    sys.stdout.write(f"\r{int(percent)} | {progress_size / (1024.**2):.2f} MB "
                     f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed")
    sys.stdout.flush()


def download_dataset():
    source = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    target = "./data/aclImdb_v1.tar.gz"

    if os.path.exists(target):
        os.remove(target)

    if not os.path.isdir("./data/aclImdb") and not os.path.isfile("./data/aclImdb_v1.tar.gz"):
        print("Downloading...")
        urllib.request.urlretrieve(source, target, reporthook=reporthook)

    if not os.path.isdir("./data/aclImdb"):
        print("\nExtracting...")
        # when only the tar file is present
        with tarfile.open(target, "r:gz") as tar:
            tar.extractall()


def load_dataset2dataframe():
    basepath = "./data/aclImdb"
    labels = {'pos': 1, 'neg': 0}

    df = pd.DataFrame()

    with tqdm(total=50000) as progress_bar:
        for subset in ("train", "test"):
            for label in ("pos", "neg"):
                path = os.path.join(basepath, subset, label)
                for file in sorted(os.listdir(path)):
                    with open(os.path.join(path, file), 'r', encoding="utf-8") as infile:
                        txt = infile.read()

                    if version.parse(pd.__version__) >= version.parse("1.3.2"):
                        x = pd.DataFrame(
                            [[txt, labels[label]]], columns=["review", "sentiment"])
                        df = pd.concat([df, x], ignore_index=False)
                    else:
                        df = df.append([[txt, labels[label]]], ignore_index=True)

                    progress_bar.update(1)  # update progress bar 1 file at a time

    df.columns = ["text", "label"]

    np.random.seed(0)
    df = df.reindex(np.random.permutation(df.index))   # shuffle the dataset

    print("Dataset shape:", df.shape)
    print("Dataset columns:", df.columns)
    print("Dataset head:", df.head())
    print("Class distribution:", np.bincount(df["label"].values))

    return df


def partition_dataset(df):
    df_shuffled = df.sample(frac=1, random_state=1).reset_index()

    train_size = 35_000

    train_df = df_shuffled.iloc[:train_size]
    val_df = df_shuffled.iloc[train_size:40_000]
    test_df = df_shuffled.iloc[40_000:]

    if not os.path.exists("./data"):
        print("Creating data folder...")
        os.makedirs("./data")

    train_df.to_csv(os.path.join("./data", "train.csv"), index=False, encoding="utf-8")
    val_df.to_csv(os.path.join("./data", "validation.csv"), index=False, encoding="utf-8")
    test_df.to_csv(os.path.join("./data", "test.csv"), index=False, encoding="utf-8")

def get_dataset():
    files = ("train.csv", "validation.csv", "test.csv")
    downloaded = True

    for file in files:
        if not os.path.exists(file):
            downloaded = False

    if downloaded is False:
        download_dataset()
        df = load_dataset2dataframe()
        partition_dataset(df)

    train_data = pd.read_csv(os.path.join("data", "train.csv"))
    val_data = pd.read_csv(os.path.join("data", "validation.csv"))
    test_data = pd.read_csv(os.path.join("data", "test.csv"))

    return train_data, val_data, test_data


def tokenization():
    imdb_dataset = load_dataset(
        "csv",
        data_files={
            "train": os.path.join("data", "train.csv"),
            "validation": os.path.join("data", "validation.csv"),
            "test": os.path.join("data", "test.csv"),
        },
    )
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_text(batch):
        return tokenizer(batch["text"], truncation=True, padding=True)

    imdb_tokenized = imdb_dataset.map(tokenize_text, batched=True, batch_size=None)
    imdb_tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return imdb_tokenized


class IMDBDataset(Dataset):
    def __init__(self, dataset_dict, partition_key="train"):
        self.partition = dataset_dict[partition_key]

    def __getitem__(self, idx):
        return self.partition[idx]

    def __len__(self):
        return self.partition.num_rows

class IMDBDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=12, num_workers=0):
        super().__init__()
        self.imdb_tokenized = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        if not os.path.exists(os.path.join("data", "train.csv")):
            get_dataset()
        self.imdb_tokenized = tokenization()

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = IMDBDataset(self.imdb_tokenized, "train")
            self.val_dataset = IMDBDataset(self.imdb_tokenized, "validation")

        if stage == "test" or stage is None:
            self.test_dataset = IMDBDataset(self.imdb_tokenized, "test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

# count number of trainable parameters
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)