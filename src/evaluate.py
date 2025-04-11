# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.jit import RecursiveScriptModule

# other libraries
from typing import Final
import re

# own modules
om src.data import load_data, load_embeddings
from src.utils import set_seed, load_model
from src.train_functions import t_step, t_step_nersa

# static variables
DATA_PATH: Final[str] = "data"
EMBEDINGS_PATH: Final[str] = "embeddings"

# set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main(name) -> None:
    """
    This function is the main program.
    """
    test_data: DataLoader
    w2v_model = load_embeddings(EMBEDINGS_PATH)

    _, _, test_data = load_data(w2v_model = w2v_model,
                                save_path=DATA_PATH, 
                                batch_size=64, 
                                shuffle=True, 
                                drop_last=False, 
                                num_workers=4)
    
    match = re.search(r"glove_50d_([^_]+)_", name)
    modo = match.group(1)
    
    model: RecursiveScriptModule = load_model(f"{name}.pt").to(device)

    if modo == "NERSA":
        test_ner, test_sa = t_step_nersa(model, test_data, device)
    else:
        test = t_step(modo, model, test_data, device)
        


if __name__ == "__main__":
    main("best_model")
