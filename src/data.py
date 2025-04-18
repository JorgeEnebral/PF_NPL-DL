from typing import Tuple, List, Optional

# deep learning libraries
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import pipeline
from torch.nn.utils.rnn import pad_sequence

# other libraries
import os
import torch
import zipfile
import requests
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from functools import partial


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
        w2v_model,
        save_path: str,
        batch_size: int = 128,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 0,
        unique_test: bool = False
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    
    """
    # Checking if provided directory exist and if not create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        download_data(save_path)
        
    if unique_test:
        unique_test_df = pd.read_json(os.path.join(save_path, f"unique_test.json"), orient="records", lines=True)
        unique_test_dataset = AlertsDataset(unique_test_df)
        collate_with_w2v = partial(collate_fn, w2v_model=w2v_model)
        unique_test_loader = DataLoader(unique_test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_with_w2v)
        return unique_test_loader, None, None
        
    train_df = pd.read_json(os.path.join(save_path, f"train.json"), orient="records", lines=True)
    val_df = pd.read_json(os.path.join(save_path, f"validation.json"), orient="records", lines=True)
    test_df = pd.read_json(os.path.join(save_path, f"test.json"), orient="records", lines=True)

    # Create datasets
    train_dataset = AlertsDataset(train_df)
    val_dataset = AlertsDataset(val_df)
    test_dataset = AlertsDataset(test_df)
    
    # Create dataloaders
    collate_with_w2v = partial(collate_fn, w2v_model=w2v_model)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_with_w2v)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_with_w2v)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_with_w2v)
    
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
        df.to_json(os.path.join(path, f"{part}.json"), orient="records", lines=True)
        
        return None
        
        
    sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    dataset = load_dataset("conll2003", trust_remote_code=True)
    
    process_dataset(sentiment_analyzer, dataset, "train")
    process_dataset(sentiment_analyzer, dataset, "validation")
    process_dataset(sentiment_analyzer, dataset, "test")
    
    return None
    

def load_embeddings(path):
    glove_zip_url = 'https://nlp.stanford.edu/data/glove.twitter.27B.zip'
    glove_zip_path = os.path.join(path, 'glove.twitter.27B.zip')
    glove_txt_file = os.path.join(path, 'glove.twitter.27B.50d.txt')
    word2vec_output_file = os.path.join(path, 'glove.twitter.27B.50d.word2vec.txt')

    # Crear directorio si no existe
    if not os.path.exists(path):
        os.makedirs(path)

    # Descargar el archivo zip si no existe
    if not os.path.exists(glove_zip_path):
        print("DESCARGANDO EMBEDDINGS DE GLOVE...")
        response = requests.get(glove_zip_url, stream=True)
        with open(glove_zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    # Descomprimir el archivo si no existe el archivo específico
    if not os.path.exists(glove_txt_file):
        print("DESCOMPRIMIENDO ARCHIVO...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
            zip_ref.extractall(path)

    # Convertir a formato Word2Vec si no existe
    if not os.path.exists(word2vec_output_file):
        print("CONVIRTIENDO A FORMATO WORD2VEC...")
        glove2word2vec(glove_txt_file, word2vec_output_file)

    # Cargar el modelo
    print("CARGANDO MODELO DE WORD2VEC...")
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    print("MODELO CARGADO CON ÉXITO.")
    return w2v_model
   
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
    return ner_map[ner_tag]


def map_sa_tags(sa_tag):

    sa_map = {
        0: "Negative", 
        1: "Neutral",
        2: "Positive", 
    }
    return sa_map[sa_tag]


def word2idx(embedding_model, tweet: List[str], ner: List[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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
    indexes = []
    ner_idx = []
    for idx, word in enumerate(tweet):
        if word in embedding_model.key_to_index:
            indexes.append(embedding_model.key_to_index[word]+1)  # Para el vector de padding en la posición 0
            if ner is not None:
                ner_idx.append(ner[idx])
    return torch.tensor(indexes, dtype=torch.long), torch.tensor(ner_idx, dtype=torch.long)


def collate_fn(batch: List[Tuple[List[str], List[int], int]], w2v_model):
    """
    Prepares and returns a batch for training/testing in a torch model.

    Args:
        batch (List[Tuple[List[str], List[int], int]]): A list of tuples, where each tuple contains a
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
    labels_ner = []
    labels_sa = []

    for text, label_ner, label_sa in batch:
        indexes, lab_ner = word2idx(w2v_model, text, label_ner)
        if len(indexes) > 0:
            indexes_txt.append(indexes)
            labels_ner.append(lab_ner) # Para el vector de padding en la posición 0
            labels_sa.append(label_sa)
    
    # Ordenar por longitud descendente
    lengths = torch.tensor([len(seq) for seq in indexes_txt], dtype=torch.long)
    sorted_data = sorted(zip(indexes_txt, labels_ner, labels_sa, lengths), key=lambda x: x[3], reverse=True)
    texts_indexes, labels_ner, labels_sa, lengths = zip(*sorted_data)

    texts_padded = pad_sequence(texts_indexes, batch_first=True, padding_value=0)
    labels_ner_padded = pad_sequence(labels_ner, batch_first=True, padding_value=-1)
    labels_sa = torch.tensor(labels_sa, dtype=torch.int32)
    lengths = torch.tensor(lengths, dtype=torch.int32)
    
    return texts_padded, labels_ner_padded, labels_sa, lengths
    
    
# if __name__ == "__main__":
    
    # python -m src.data para cargarlos
    
    # DATA_PATH = "data"
    # EMBEDINGS_PATH = "embeddings"

    # w2v_model = load_embeddings(EMBEDINGS_PATH) 
    #  dat_t, dat_v, dat_te = load_data(w2v_model=w2v_model, save_path=DATA_PATH)