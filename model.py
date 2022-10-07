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
        device: torch.device
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.device = device
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.dropout=dropout
        self.bidirectional=bidirectional
        self.num_class=num_class
        self.gru=torch.nn.GRU(input_size=embeddings.size()[1],hidden_size=hidden_size,
        num_layers=num_layers,dropout=dropout,
        bidirectional=bidirectional,batch_first=True)
        if self.bidirectional:
            self.output_layer = torch.nn.Linear(in_features=2*hidden_size,out_features=num_class)
        else:
            self.output_layer = torch.nn.Linear(in_features=hidden_size,out_features=num_class)
        # self.normalize=torch.nn.Softmax(dim=1)

    @property
    def encoder_output_size(self) -> int:
        return self.num_class

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        batch_inputs = self.embed(batch)
        batch_size = batch.size()[0]
        seq_len = batch.size()[1]
        D=1
        if self.bidirectional:
            D=2
        h_0 = torch.zeros(D*self.num_layers,batch_size,self.hidden_size,
        device = self.device, requires_grad=True)
        out, _ = self.gru(batch_inputs, h_0)
        outputs = self.output_layer(out[:,seq_len-1,:])
        # outputs = self.normalize(outputs)
        # outputs shape: batch_size * num_class
        outputs_dict = {"prediction": outputs}
        return outputs_dict

class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        batch_inputs = self.embed(batch)
        batch_size = batch.size()[0]
        seq_len = batch.size()[1]
        D=1
        if self.bidirectional:
            D=2
        h_0 = torch.zeros(D*self.num_layers,batch_size,self.hidden_size,
        device = self.device, requires_grad=True)
        out, _ = self.gru(batch_inputs, h_0)
        outputs = self.outputlayer(out)
        # outputs shape: batch_size * seq_len * num_class
        outputs_dict = {"prediction": outputs}
        return outputs_dict
