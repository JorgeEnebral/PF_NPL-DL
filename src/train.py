    # deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import os


# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data, word2idx
from src.models import NerSaModel
from src.train_functions import train_step, val_step, t_step
from src.utils import set_seed, save_model, parameters_to_double

# static variables
DATA_PATH: Final[str] = "data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)

def main() -> None:
    """
    This function is the main program for training.
    """
    # parametros
    num_epochs = 100
    hidden_size = 128
    hidden_layers = 3
    lr = 0.001
    batch_size = 128
    dropout = 0.2


    # creacion del modelo de embeddings
    
    glove_input_file = '../embeddings/glove.twitter.27B.50d.txt'
    word2vec_output_file = '../embeddings/glove.twitter.27B.50d.word2vec.txt'
    if not os.path.exists(word2vec_output_file):
        print("CREANDO EL WORD2VEC")
        glove2word2vec(glove_input_file, word2vec_output_file)

    print("OBTENCION DEL MODELO DE WORD2VEC")
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    # pesos de los embeddings
    embedding_weights = w2v_model.vectors

    # Dataloaders
    print("OBTENCION DE LOS DATALOADERS")
    train_data, val_data, _ = load_data(save_path=DATA_PATH, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, w2vmodel=w2v_model)
    
    # modelo de prediccion
    print("GENERACION DEL MODELO")
    model = NerSaModel(embedding_weights=embedding_weights, ).to(device)
    parameters_to_double(model)

    print("GENERACION DE LOS OPTIMIZADORES")
    optimizer_ner = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004)  # Parámetros poner los usados por ner
    optimizer_sa = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0004)  # Parámetros poner los usados por sa
    
    print("CREACION DE LAS FUNCIONES DE PERDIDA")
    loss_ner = torch.nn.CrossEntropyLoss()
    loss_sa = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)  NECESARIO?
    writer = SummaryWriter()
    
    print("ENTRENANDO...")
    for epoch in tqdm(range(num_epochs)):
        train_step(model, train_data, loss_ner, loss_sa, optimizer_ner, optimizer_sa, writer, epoch, device)
        val_step(model, val_data, loss_ner, loss_sa, writer, epoch, device)

    print("ENTRENAMIENTO COMPLETADO")

    txt2save: str = f"glove_27B_50d_{num_epochs}_{hidden_size}_{hidden_layers}_{lr}_{batch_size}_{dropout}"
    save_model(model, txt2save)
    print("MODELO GUARDADO")

    writer.close()


if __name__ == "__main__":
    main()
