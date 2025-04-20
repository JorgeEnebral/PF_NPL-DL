# deep learning libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os


# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data, load_embeddings
from src.models import NerSaModel
from src.train_functions import train_step_sa, val_step_sa, t_step_sa, train_step_ner, val_step_ner, t_step_ner, train_step_nersa, val_step_nersa, t_step_nersa
from src.utils import set_seed, save_model, parameters_to_double

# static variables
DATA_PATH: Final[str] = "data"
EMBEDINGS_PATH: Final[str] = "embeddings"
TOKENIZERS_PARALLELISM = False

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

import json
import torch
from collections import Counter

def weights(train_path="data/train.json"):
    ner_counter = Counter()
    sa_counter = Counter()
    
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # contar NER
            ner_counter.update(data['ner'])
            
            # contar SA
            sa_counter.update([data['sa']])

    # total clases
    ner_classes = 9
    sa_classes = 3
    
    # crear listas de frecuencias
    ner_freq = torch.tensor([ner_counter.get(i, 0) for i in range(ner_classes)], dtype=torch.float)
    sa_freq = torch.tensor([sa_counter.get(i, 0) for i in range(sa_classes)], dtype=torch.float)

    # evitar división por cero
    ner_freq[ner_freq == 0] = 1
    sa_freq[sa_freq == 0] = 1

    # calcular pesos inversamente proporcionales a la frecuencia
    weights_ner = 1.0 / ner_freq
    weights_sa = 1.0 / sa_freq

    # normalizar (suma 1)
    weights_ner = weights_ner / weights_ner.sum()
    weights_sa = weights_sa / weights_sa.sum()
    
    return weights_ner, weights_sa


def main() -> None:
    """
    This function is the main program for training.
    """
    # parametros
    num_epochs = 30
    hidden_size = 32
    hidden_layers = 1
    lr_sa = 0.001
    lr_ner = 0.01
    w_dc_sa = 0.0
    w_dc_ner = 0.0
    batch_size = 128
    dropout = 0.0
    modo = "NERSA" # ["NER", "SA", "NERSA"]
    loss_ponderation = torch.tensor([0.35, 0.65]) # [NER, SA]

    # Dataloaders
    print("OBTENCION DE LOS DATALOADERS")
    emb_dim = 50 # [25, 50, 100, 200]
    w2v_model = load_embeddings(EMBEDINGS_PATH, emb_dim)
    embedding_weights = torch.tensor(w2v_model.vectors)
    
    padding_vector = torch.zeros((1, embedding_weights.shape[1]))
    embedding_weights = torch.cat((padding_vector, embedding_weights), dim=0) 
    
    train_data, val_data, test_data = load_data(w2v_model = w2v_model,
                                                save_path=DATA_PATH, 
                                                batch_size=batch_size, 
                                                shuffle=True, 
                                                drop_last=False, 
                                                num_workers=4)
    
    # GENERACION DEL MODELO
    model = NerSaModel(embedding_weights, hidden_size, hidden_layers, l_pond=loss_ponderation, dropout=dropout, mode=modo).to(device)
    model.lstm.flatten_parameters() 
    parameters_to_double(model)
    
    optimizer_sa = torch.optim.Adam(model.parameters(), lr=lr_sa, weight_decay=w_dc_sa)
    optimizer_ner = torch.optim.Adam(model.parameters(), lr=lr_ner, weight_decay=w_dc_ner)
    weights_ner, weights_sa = weights()
    loss_ner = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=weights_ner.to(device))
    loss_sa = torch.nn.CrossEntropyLoss(weight=weights_sa.to(device)) 

    writer = SummaryWriter()
    
    print("ENTRENANDO")
    for epoch in tqdm(range(num_epochs)):
        if modo == "NERSA":
            train_step_nersa(model, train_data, loss_ner, loss_sa, optimizer_ner, optimizer_sa, writer, epoch, device)
            val_step_nersa(model, val_data, loss_ner, loss_sa, writer, epoch, device)
        elif modo == "NER":
            train_step_ner(model, train_data, loss_ner, optimizer_ner, writer, epoch, device)
            val_step_ner(model, val_data, loss_ner, writer, epoch, device)
        else:
            train_step_sa(model, train_data, loss_sa, optimizer_sa, writer, epoch, device)
            val_step_sa(model, val_data, loss_sa, writer, epoch, device)
            
    print("ENTRENAMIENTO COMPLETADO\n")

    if modo == "NERSA":
        t_step_nersa(model, test_data, loss_ner, loss_sa, device)
    elif modo == "NER":
        t_step_ner(model, test_data, loss_ner, device)
    else:
        t_step_sa(model, test_data, loss_sa, device)
    
    txt2save: str = f"glove_{emb_dim}d_{modo}_{num_epochs}_{batch_size}"
    save_model(model, txt2save)
    
    print("\nMODELO GUARDADO")

    writer.close()


if __name__ == "__main__":
    main()
