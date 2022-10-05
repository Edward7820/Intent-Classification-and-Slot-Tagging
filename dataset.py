from typing import List, Dict
import torch
from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        batch_data = dict()
        inputs_text=[(sample['text'].split(' ')) for sample in samples]
        batch_data['text']=torch.tensor(self.vocab.encode_batch(inputs_text,to_len=self.max_len))
        batch_data['intent']=torch.tensor([self.label_mapping[sample['intent']] for sample in samples])
        return batch_data

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError

'''
For debug:
from typing import Dict, List
import torch
from dataset import SeqClsDataset
import pickle
from utils import Vocab

data=list()
data.append({"text": "how long should i cook steak for","intent": "cook_time","id": "eval-0"})
data.append({"text": "please tell me how much money i have in my bank accounts","intent": "balance","id": "eval-1"})
data.append({"text": "what is the gas level in my gas tank","intent": "gas","id": "eval-2"})
with open("cache/intent/vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)
label_mapping = dict()
label_mapping["cook_time"]=0
label_mapping["balance"]=1
label_mapping["gas"]=2
max_len=10
dataset=SeqClsDataset(data, vocab, label_mapping, max_len)
print(dataset.collate_fn(data[0:3]))
'''