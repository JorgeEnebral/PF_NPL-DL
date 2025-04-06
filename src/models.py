# deep learning libraries
import torch
import torch.nn as nn

# other libraries
import math
            
# ATENCIÓN: DE MOMENTO SOLO HE DISEÑADO LA ¡SA! FALTA LA ¡NER! Y CONECTARLASSSS!!!!!
# ATENCIÓN
# ATENCIÓN
# ATENCIOÓN
# ATENCIÓNNNNNNN 
# ÎÎÎÎÎÎÎÎÎÎÎÎÎ
class NerSaModel(torch.nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, 
                 hidden_size: int, 
                 hidden_layers: int, 
                 dropout: float, 
                 output_dim: int=3) -> None:
        """
        This method is the constructor of the class.

        Args:
            hidden_size: hidden size of the RNN layers
        """
        super(NerSaModel, self).__init__()
        # TODO: Create an embedding layer with the given pre-trained weights, use the Embedding.from_pretrained function
        _, embedding_dim = embedding_weights.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        # capa LSTM BIDIRECCIONAL!!!! 
        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=hidden_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, output_dim)


    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        This method is the forward pass of the model.

        Args:
            inputs: inputs tensor. Dimensions: [batch, number of past days, 24].

        Returns:
            output tensor. Dimensions: [batch, 24].
        """
        # convertimos los inputs a embeddings
        embedded = self.embedding(inputs)

        # para que detecte la longitud de las frases sin hacer calculos innecesarios con los paddings (que son 0)
        # lengths.cpu indica la longitud real de la oración, sin tener en cuenta el padding
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=True)

        # capa de la LSTM
        packed_out, (hidden, _) = self.lstm(packed)

        # añadimos dropout para que no haga overfittng
        output = self.dropout(hidden[-1])

        # fully connected
        return self.fc(output)