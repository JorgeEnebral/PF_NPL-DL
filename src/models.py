# deep learning libraries
import torch
import torch.nn as nn

# other libraries
import math
            
            
class NerSaModel(torch.nn.Module):
    def __init__(self, embedding_weights: torch.Tensor, 
                 hidden_size: int, 
                 hidden_layers: int, 
                 dropout: float = 0.0, 
                 mode: str = "NERSA") -> None:
        """
        This method is the constructor of the class.

        Args:
            mode: "NER", "SA" "NERSA"
        """
        super(NerSaModel, self).__init__()
        
        self.mode = mode
        self.output_dims = {"NER": 9,
                            "SA": 3}

        _, embedding_dim = embedding_weights.shape
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True)

        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=hidden_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)

        if self.mode == "NERSA":
            self.fc_ner = nn.Linear(hidden_size, self.output_dims["NER"])
            self.fc_sa = nn.Linear(hidden_size, self.output_dims["SA"])
        else:
            self.fc = nn.Linear(hidden_size, self.output_dims[self.mode])


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