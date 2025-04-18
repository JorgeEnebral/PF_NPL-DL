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

def main() -> None:
    """
    This function is the main program for training.
    """
    # parametros
    num_epochs = 30
    hidden_size = 32
    hidden_layers = 1
    lr_sa = 0.001
    lr_ner = 0.001
    w_dc_sa = 0.001
    w_dc_ner = 0.001
    batch_size = 128
    dropout = 0.2
    modo = "NERSA" # ["NER", "SA", "NERSA"]
    loss_ponderation = torch.tensor([0.4, 0.4, 0.2]) # [0.1, 0.9]

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
    model.lstm.flatten_parameters()  # Lo recomienda chat
    parameters_to_double(model)
    
    optimizer_sa = torch.optim.Adam(model.parameters(), lr=lr_sa, weight_decay=w_dc_sa)
    optimizer_ner = torch.optim.Adam(model.parameters(), lr=lr_ner, weight_decay=w_dc_ner)
    loss_ner_out = torch.nn.CrossEntropyLoss(ignore_index=-1)
    loss_ner_ent = torch.nn.CrossEntropyLoss(ignore_index=-1)
    loss_sa = torch.nn.CrossEntropyLoss() 

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)  NECESARIO?
    writer = SummaryWriter()
    
    print("ENTRENANDO")
    for epoch in tqdm(range(num_epochs)):
        if modo == "NERSA":
            train_step_nersa(model, train_data, loss_ner_out, loss_ner_ent, loss_sa, optimizer_ner, optimizer_sa, writer, epoch, device)
            val_step_nersa(model, val_data, loss_ner_out, loss_ner_ent, loss_sa, writer, epoch, device)
        elif modo == "NER":
            train_step_ner(model, train_data, loss_ner_out, loss_ner_ent, optimizer_ner, writer, epoch, device)
            val_step_ner(model, val_data, loss_ner_out, loss_ner_ent, writer, epoch, device)
        else:
            train_step_sa(model, train_data, loss_sa, optimizer_sa, writer, epoch, device)
            val_step_sa(model, val_data, loss_sa, writer, epoch, device)
            
    print("ENTRENAMIENTO COMPLETADO\n")

    if modo == "NERSA":
        t_step_nersa(model, test_data, loss_ner_out, loss_ner_ent, loss_sa, device)
    elif modo == "NER":
        t_step_ner(model, test_data, loss_ner_out, loss_ner_ent, device)
    else:
        t_step_sa(model, test_data, loss_sa, device)
    
    txt2save: str = f"glove_{emb_dim}d_{modo}_{num_epochs}_{batch_size}"
    save_model(model, txt2save)
    
    print("\nMODELO GUARDADO")

    writer.close()


if __name__ == "__main__":
    main()
