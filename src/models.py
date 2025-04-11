from typing import Optional, Tuple

# deep learning libraries
import torch
import torch.nn as nn

            
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

        # PROBAR SI CUADRA
        _, embedding_dim = embedding_weights.shape                                  
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_size=hidden_size, num_layers=hidden_layers, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        
        self.fc_ner = nn.Linear(2 * hidden_size, self.output_dims["NER"]) if mode == "NERSA" else nn.Identity()
        self.fc_sa = nn.Linear(2 * hidden_size, self.output_dims["SA"]) if mode == "NERSA" else nn.Identity()
        self.fc = nn.Linear(2 * hidden_size, self.output_dims[self.mode])


    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        packed_output, (hidden, _) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        hiddens_concat = torch.cat((hidden[-1], hidden[-2]), dim=-1)

        if self.mode == "NER":
            output_ner = self.fc(outputs)
            return output_ner, None
        elif self.mode == "SA":
            output_sa = self.fc(hiddens_concat)
            return output_sa, None
        elif self.mode == "NERSA":
            output_ner = self.fc_ner(outputs)
            output_sa = self.fc_sa(hiddens_concat)
            return output_ner, output_sa
        return None, None