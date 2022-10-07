from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from dataset import SeqTaggingClsDataset
from model import SeqTagger
import pickle
from utils import Vocab

data=list()
data.append({"tokens":["i","have","three","people","for","august","seventh"],
"tags":["O","O","B-people","I-people","O","B-date","O"],"id": "train-0"})
data.append({"tokens": ["do","you","have","highchairs","for","my","4","kids"],
"tags":["O","O","O","O","O","O","O","O"],"id": "train-1"})
with open("cache/intent/vocab.pkl", "rb") as f:
    vocab: Vocab = pickle.load(f)
label_mapping = dict()
label_mapping["O"]=0
label_mapping["B-people"]=1
label_mapping["I-people"]=2
label_mapping["B-date"]=3
max_len=10
dataset=SeqTaggingClsDataset(data, vocab, label_mapping, max_len)
print(dataset.collate_fn(data[0:2]))