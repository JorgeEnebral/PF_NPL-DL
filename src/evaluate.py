# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final

# own modules
from src.data import load_data
from src.utils import set_seed, load_model
from src.train_functions import t_step

# static variables
DATA_PATH: Final[str] = "data"
NUM_CLASSES: Final[int] = 10

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program.
    """
    test_data: DataLoader
    _, _, test_data, mean, std = load_data(DATA_PATH, batch_size=32, num_workers=4)

    model: RecursiveScriptModule = load_model("best_model.pt").to(device)

    test_mae = t_step(model, test_data, mean, std, device)


if __name__ == "__main__":
    main()