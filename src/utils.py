# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random
from typing import Optional, Tuple


class Accuracy:
    """
    This class is the accuracy object.

    Attr:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self, mode: str) -> None:
        """
        Constructor of Accuracy class.
        Initializes correct and total to zero.
        """
        self.mode = mode
        
        self.correct = 0
        self.total = 0
        
        self.correct_out = 0
        self.total_out = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions for SA: [batch, num_classes]
                Dimensions for NER: [batch, seq_len, num_classes]
            labels: true labels. For NER: [batch, seq_len]
        """
        if self.mode == "SA":
            predictions = logits.argmax(dim=1).type_as(labels)
            self.correct += int((predictions == labels).sum().item()) # bs
            self.total += labels.shape[0]

        else:  # NER o NERSA
            predictions = logits.argmax(dim=2).type_as(labels)
            
            mask_out = labels == 0
            mask_ent = (labels != -1) & (labels != 0)
            
            self.correct += int(((predictions == labels) & mask_ent).sum().item())
            self.total += int(mask_ent.sum().item())
            
            self.correct_out += int(((predictions == labels) & mask_out).sum().item())
            self.total_out += int(mask_out.sum().item())
            
        return None

    def compute(self) -> Tuple[float, Optional[float]]:
        """
        Returns the accuracy value.
        """
        if self.mode == "SA":
            return self.correct / self.total if self.total > 0 else 0.0, None
        else:
            labels = self.correct / self.total if self.total > 0 else 0.0
            out = self.correct_out / self.total_out if self.total_out > 0 else 0.0
            return labels, out


@torch.no_grad()
def parameters_to_double(model: torch.nn.Module) -> None:
    """
    This function transforms the model parameters to double.

    Args:
        model: pytorch model.
    """
    for parameter in model.parameters():
        parameter.data = parameter.data.double()


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None


def load_model(name: str) -> RecursiveScriptModule:
    """
    This function is to load a model from the 'models' folder.

    Args:
        name: name of the model to load.

    Returns:
        model in torchscript.
    """

    # define model
    model: RecursiveScriptModule = torch.jit.load(f"models/{name}.pt")

    return model


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed: seed number to fix radomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
