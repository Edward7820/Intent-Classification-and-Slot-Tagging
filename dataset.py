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
        batch_data['id']=[sample['id'] for sample in samples]
        if 'text' in samples[0].keys():
            inputs_text=[(sample['text'].split(' ')) for sample in samples]
            batch_data['text']=torch.tensor(self.vocab.encode_batch(inputs_text,to_len=self.max_len))
        if 'intent' in samples[0].keys():
            batch_data['intent']=torch.tensor([self.label_mapping[sample['intent']] for sample in samples])
        return batch_data

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    @property
    def ignore_idx(self) -> int:
        return -100

    @property
    def token_nums(self) -> dict:
        id2nums = dict()
        for instance in self.data:
            id2nums[instance['id']] = len(instance['tokens'])
        return id2nums

    def collate_fn(self, samples):
        batch_data = dict()
        batch_data['id']=[sample['id'] for sample in samples]
        if 'tokens' in samples[0].keys():
            inputs_token=[sample['tokens'] for sample in samples]
            batch_data['tokens']=torch.tensor(self.vocab.encode_batch(inputs_token,to_len=self.max_len))
        if 'tags' in samples[0].keys():
            inputs_tag = list()
            for sample in samples:
                all_tags=[self.label_mapping[tag] for tag in sample['tags']]
                padded_tags=all_tags[:self.max_len]+[self.ignore_idx]*max(0 , self.max_len - len(all_tags))
                inputs_tag.append(padded_tags)
                # print(inputs_tag)
            batch_data['tags']=torch.tensor(inputs_tag)
        return batch_data