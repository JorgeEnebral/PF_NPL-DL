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
from src.train_functions import train_step, val_step, t_step, train_step_nersa, val_step_nersa, t_step_nersa
from src.utils import set_seed, save_model, parameters_to_double

# static variables
DATA_PATH: Final[str] = "data"
EMBEDINGS_PATH: Final[str] = "embeddings"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def main() -> None:
    """
    This function is the main program for training.
    """
    # parametros
    num_epochs = 5
    hidden_size = 64
    hidden_layers = 2
    lr = 0.001
    batch_size = 128
    dropout = 0.0
    modo = "SA" # ["NER", "SA", "NERSA"]

    # Dataloaders
    print("OBTENCION DE LOS DATALOADERS")
    w2v_model = load_embeddings(EMBEDINGS_PATH)
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
    model = NerSaModel(embedding_weights, hidden_size, hidden_layers, mode=modo).to(device)
    model.lstm.flatten_parameters()  # Lo recomienda chat
    parameters_to_double(model)

    optimizer_ner = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004)
    loss_ner = torch.nn.CrossEntropyLoss()
    
    optimizer_sa = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004)
    loss_sa = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)  NECESARIO?
    writer = SummaryWriter()
    
    print("ENTRENANDO")
    for epoch in tqdm(range(num_epochs)):
        if modo == "NERSA":
            train_step_nersa(model, train_data, loss_ner, loss_sa, optimizer_ner, optimizer_sa, writer, epoch, device)
            val_step_nersa(model, val_data, loss_ner, loss_sa, writer, epoch, device)
        else:
            # Da igual el nombre de las variables loss_ner y optimizer_ner
            train_step(modo, model, train_data, loss_ner, optimizer_ner, writer, epoch, device)
            val_step(modo, model, val_data, loss_ner, writer, epoch, device)

    print("ENTRENAMIENTO COMPLETADO\n")

    if modo == "NERSA":
        t_step_nersa(model, test_data, device)
    else:
        t_step(modo, model, test_data, device)
    
    txt2save: str = f"glove_50d_{modo}_{num_epochs}_{batch_size}_{hidden_size}_{hidden_layers}_{lr}_{dropout}"
    save_model(model, txt2save)
    
    print("\nMODELO GUARDADO")

    writer.close()


if __name__ == "__main__":
    main()
