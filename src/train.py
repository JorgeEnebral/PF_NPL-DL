    # deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm
from typing import Final

# own modules
from src.data import load_data
from src.models import MyModel
from src.train_functions import train_step, val_step, t_step
from src.utils import set_seed, save_model

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
    
    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0004)
    loss_fn = torch.nn.L1Loss()  # MAE
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.2)
    writer = SummaryWriter()
    
    num_epochs = 100
    
    for epoch in tqdm(range(num_epochs)):
        train_step(model, train_data, loss_fn, optimizer, writer, epoch, device)
        val_step(model, val_data, loss_fn, scheduler, writer, epoch, device)
    
    # Final evaluation
    t_step(model, test_data, device)
    save_model(model, "best_model")
    writer.close()


if __name__ == "__main__":
    main()
