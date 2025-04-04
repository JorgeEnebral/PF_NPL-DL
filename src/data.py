from typing import Tuple, List

# deep learning libraries
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import pipeline

# other libraries
import os
import random

# own modules
from src.utils import set_seed
set_seed(42)


class AlertsDataset(Dataset):
    """
    A PyTorch Dataset for the AlertsDataset dataset.
    """
    def __init__(self, df):
        """
        Initializes the TweepFakeDataset with the given file path.
        """
        self.texts = df["tokens"]
        self.ner_targets = df["ner"]
        self.sa_targets = df["sa"]

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.sa_targets)

    def __getitem__(self, idx: int) -> Tuple[List[str], List[int], int]:
        """
        Returns the embedded tensor and target for the text at the specified index.
        """
        return self.texts[idx], self.ner_targets[idx], self.sa_targets[idx]


def load_data(
    save_path: str,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    
    """
    # Checking if provided directory exist and if not create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        download_data(save_path)
        
    train_df = pd.read_csv(f"{save_path}/train.csv", sep="\t")
    val_df = pd.read_csv(f"{save_path}/val.csv", sep="\t")
    test_df = pd.read_csv(f"{save_path}/test.csv", sep="\t")

    # Create datasets
    train_dataset = AlertsDataset(train_df)
    val_dataset = AlertsDataset(val_df)
    test_dataset = AlertsDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def download_data(path) -> None:
    """"""
    def process_dataset(sentiment_analyzer, dataset, part):
        """"""        
        data_tokens = [sent["tokens"] for sent in dataset[part]]
        data_ner_tags= [sent["ner_tags"] for sent in dataset[part]]
        data_sa = [int(sentiment_analyzer(" ".join(sent))["label"][-1]) for sent in data_tokens]
        
        df = pd.DataFrame({
            "tokens": data_tokens,
            "ner": data_ner_tags,
            "sa": data_sa
        })
        df.to_csv(os.path.join(path, f"{part}.csv"), index=False, sep="\t")
        
        return None
        
        
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    dataset = load_dataset("conll2003")
    
    process_dataset(sentiment_analyzer, dataset, "train")
    process_dataset(sentiment_analyzer, dataset, "validation")
    process_dataset(sentiment_analyzer, dataset, "test")
    
    return None
    
    
    
    
    
    