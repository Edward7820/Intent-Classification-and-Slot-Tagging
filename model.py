from typing import Dict, List

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.num_class=num_class
        self.rnn=torch.nn.RNN(input_size=embeddings.size()[1],hidden_size=hidden_size,
        num_layers=num_layers,nonlinearity='tanh',dropout=dropout,
        bidirectional=bidirectional)
        self.output_layer=torch.nn.Linear(in_features=hidden_size,out_features=num_class)

    @property
    def encoder_output_size(self) -> int:
        return self.num_class

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        batch_size=batch.size()


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
