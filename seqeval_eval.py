from seqeval.scheme import IOB2
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report

import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab
import csv

def main(args):
    global y_pred
    global y_true
    y_pred=list()
    y_true=list()
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)

    def collate_fn(samples: List[Dict]) -> Dict:
        return dataset.collate_fn(samples)
    dataloader = DataLoader(dataset,batch_size=args.batch_size,
    collate_fn=collate_fn, pin_memory=True, shuffle=False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        args.device,
    )
    model = model.to(args.device)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.eval()

    for test_batch in dataloader:
        test_batch['tokens'] = test_batch['tokens'].to(args.device)
        output = (model(test_batch['tokens']))['prediction']
        # output shape: batch_size * max_len * num_classes
        test_batch_size = test_batch['tokens'].size()[0]
        for i in range(test_batch_size):
            y_pred.append([])
            y_true.append([])
            for j in range(args.max_len):
                if test_batch['tokens'][i][j] != vocab.pad_id:
                    y_pred[-1].append(dataset.idx2label(torch.argmax(output[i,j]).item()))
                    y_true[-1].append(dataset.idx2label(test_batch['tags'][i][j].item()))
    # print(y_pred)
    # print(y_true)
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)