from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from dataset import SeqClsDataset
from model import SeqClassifier
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
print(dataset.collate_fn(data[0:2]))
def collate_fn(samples: List[Dict]) -> Dict:
    batch_data = dict()
    inputs_text=[(sample['text'].split(' ')) for sample in samples]
    batch_data['text']=torch.tensor(dataset.vocab.encode_batch(inputs_text,to_len=dataset.max_len))
    batch_data['intent']=torch.tensor([dataset.label_mapping[sample['intent']] for sample in samples])
    return batch_data
loader = DataLoader(dataset,batch_size=2,collate_fn=collate_fn,shuffle=True)
embeddings = torch.load("cache/intent/embeddings.pt")
for epoch in range(5):
    print(epoch)
    for batch in loader:
        print(batch)

is_cuda = torch.cuda.is_available()
if (is_cuda==0):
    device = torch.device("cpu")
    print("cuda not available")
else:
    device = torch.device("cuda")
    print("cuda is used")