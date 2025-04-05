# deep learning libraries
import torch
import torch.nn as nn

# other libraries
import math
            

class NerSaModel(torch.nn.Module):
    def __init__(self) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """
        super(NerSaModel, self).__init__()


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """
        return