    # deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
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
    train_data, val_data, test_data = load_data(DATA_PATH, batch_size=128, shuffle=True, drop_last=False, num_workers=4)
    
    model = NerSaModel().to(device)
    parameters_to_double(model)

    optimizer_ner = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)  # Parámetros poner los usados por ner
    optimizer_sa = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)  # Parámetros poner los usados por sa

    loss_ner = torch.nn.CrossEntropyLoss()
    loss_sa = torch.nn.CrossEntropyLoss()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)  NECESARIO?
    writer = SummaryWriter()
    
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        train_step(model, train_data, loss_ner, loss_sa, optimizer_ner, optimizer_sa, writer, epoch, device)
        val_step(model, val_data, loss_ner, loss_sa, writer, epoch, device)
    
    # Final evaluation
    t_step(model, test_data, device)
    save_model(model, "best_model")
    writer.close()


if __name__ == "__main__":
    main()
