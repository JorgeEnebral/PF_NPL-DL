# deep learning libraries
import torch
import numpy as np
from torch.jit import RecursiveScriptModule

# other libraries
import os
import random


class Accuracy:
    """
    This class is the accuracy object.

    Attr:
        correct: number of correct predictions.
        total: number of total examples to classify.
    """

    correct: int
    total: int

    def __init__(self, modo: str) -> None:
        """
        Constructor of Accuracy class.
        Initializes correct and total to zero.
        """
        self.correct = 0
        self.total = 0
        self.modo = modo

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Updates the value of correct and total counts.

        Args:
            logits: outputs of the model.
                Dimensions for SA: [batch, num_classes]
                Dimensions for NER: [batch, seq_len, num_classes]
            labels: true labels. For NER: [batch, seq_len]
        """
        if self.modo == "SA":
            predictions = logits.argmax(dim=1).type_as(labels)
            self.correct += int((predictions == labels).sum().item())
            self.total += labels.shape[0]

        else:  # NER o NERSA
            # logits: [batch, seq_len, num_classes]
            # labels: [batch, seq_len]
            predictions = logits.argmax(dim=2).type_as(labels)

            # Flatten both to compute token-level accuracy
            mask = labels != -100  # O cualquier ignore_index que estés usando
            self.correct += int((predictions[mask] == labels[mask]).sum().item())
            self.total += int(mask.sum().item())

        return None

    def compute(self) -> float:
        """
        Returns the accuracy value.
        """
        return self.correct / self.total if self.total > 0 else 0.0


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
