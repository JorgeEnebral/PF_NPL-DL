import gensim
from gensim.models.keyedvectors import KeyedVectors
from torch.nn.utils.rnn import pad_sequence


from typing import Tuple, List

# deep learning libraries
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import pipeline

# other libraries
import os
import torch

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
        w2vmodel,
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
        
    train_df = pd.read_csv(f"{save_path}/train.csv", sep=";")
    val_df = pd.read_csv(f"{save_path}/validation.csv", sep=";")
    test_df = pd.read_csv(f"{save_path}/test.csv", sep=";")

    # Create datasets
    train_dataset = AlertsDataset(train_df)
    val_dataset = AlertsDataset(val_df)
    test_dataset = AlertsDataset(test_df)
    
    # Create dataloaders
    # he añadido el collate_fn, al cual le paso como argumento el modelo de w2v
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn(w2v_model=w2vmodel))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn(w2v_model=w2vmodel))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn(w2v_model=w2vmodel))
    
    return train_loader, val_loader, test_loader


def download_data(path) -> None:
    """"""
    def process_dataset(sentiment_analyzer, dataset, part):
        """"""        
        data_tokens = [[token.replace('"', '').replace("'", "") for token in sent["tokens"] 
                        if token.replace('"', '').replace("'", "")] for sent in dataset[part]]
        data_ner_tags= [sent["ner_tags"] for sent in dataset[part]]
        data_sa = [int(sentiment_analyzer(" ".join(sent))[0]["label"][-1]) for sent in data_tokens]
        
        df = pd.DataFrame({
            "tokens": data_tokens,
            "ner": data_ner_tags,
            "sa": data_sa
        })
        df.to_csv(os.path.join(path, f"{part}.csv"), index=False, sep=";")
        
        return None
        
        
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    
    process_dataset(sentiment_analyzer, dataset, "train")
    process_dataset(sentiment_analyzer, dataset, "validation")
    process_dataset(sentiment_analyzer, dataset, "test")
    
    return None
    
       
def map_ner_tags(ner_tag):
    # Mapa de etiquetas NER de CoNLL-2003 a las etiquetas del esquema propuesto
    ner_map = {
        0: "O",   # No es una entidad
        1: "B-PER",  # Inicio de Persona
        2: "I-PER",  # Continuación de Persona
        3: "B-ORG",  # Inicio de Organización
        4: "I-ORG",  # Continuación de Organización
        5: "B-LOC",  # Inicio de Ubicación
        6: "I-LOC",  # Continuación de Ubicación
        7: "B-MISC",  # Inicio de Entidad Miscelánea
        8: "I-MISC"   # Continuación de Entidad Miscelánea
    }
    return [ner_map[tag] for tag in ner_tag]


def map_sa_tags(sa_tag):

    sa_map = {
        0: "Negative", 
        1: "Neutral",
        2: "Positive", 
    }
    return [sa_map[tag] for tag in sa_tag]   


def word2idx(embedding_model, tweet: List[str]) -> torch.Tensor:
    """
    Converts a tweet to a list of word indices based on an embedding model.

    This function iterates through each word in the tweet and retrieves its corresponding index
    from the embedding model's vocabulary. If a word is not present in the model's vocabulary,
    it is skipped.

    Args:
        embedding_model (Any): The embedding model with a 'key_to_index' attribute, which maps words to their indices.
        tweet (List[str]): A list of words representing the tweet.

    Returns:
        torch.Tensor: A tensor of word indices corresponding to the words in the tweet.
    """
    # TODO: Complete the function according to the requirements
    indices = [
        embedding_model.key_to_index[word]
        for word in tweet
        if word in embedding_model.key_to_index
    ]
    return torch.tensor(indices)


def collate_fn(batch: List[Tuple[List[str], int]], w2v_model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepares and returns a batch for training/testing in a torch model.

    Args:
        batch (List[Tuple[List[str], int]]): A list of tuples, where each tuple contains a
                                             list of words (representing a text) and an integer label.
        word2idx (Callable): Function that converts words to indices.
        embedding_model: Word embedding model (e.g., Word2Vec, FastText, etc.).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - texts_padded (torch.Tensor): Tensor of padded word indices.
            - labels (torch.Tensor): Tensor of labels.
            - lengths (torch.Tensor): Tensor of sequence lengths.
    """

    indexes_txt = []
    clean_labels = []

    for text, label in batch:
        indexes = word2idx(w2v_model, text)
        if len(indexes) == 0:
            indexes = torch.tensor([0])
        indexes_txt.append(indexes)
        clean_labels.append(label)

    # Ordenar por longitud descendente
    lengths = torch.tensor([len(seq) for seq in indexes_txt], dtype=torch.long)
    sorted_data = sorted(zip(indexes_txt, clean_labels, lengths), key=lambda x: x[2], reverse=True)

    texts_indexes, labels_cleaned, lengths = zip(*sorted_data)

    texts_padded = pad_sequence(texts_indexes, True, 0)
    labels = torch.tensor(labels_cleaned, dtype=torch.float)
    lengths = torch.tensor(lengths, dtype=torch.long)

    return texts_padded, labels, lengths

    
    
    
if __name__ == "__main__":
    
    # python -m src.data para cargarlos
    path = "data"
    dat_t, dat_v, dat_te = load_data(path)