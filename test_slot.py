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

    prediction = list()
    prediction.append(['id','tags'])
    for test_batch in dataloader:
        test_batch['tokens'] = test_batch['tokens'].to(args.device)
        output = (model(test_batch['tokens']))['prediction']
        # output shape: batch_size * max_len * num_classes
        test_batch_size = test_batch['tokens'].size()[0]
        for i in range(test_batch_size):
            pred_tags = ''
            for j in range(args.max_len):
                if test_batch['tokens'][i][j] != vocab.pad_id:
                    if pred_tags != '':
                        pred_tags += ' '
                    pred_tags += dataset.idx2label(torch.argmax(output[i,j]).item())
            prediction.append([test_batch['id'][i],pred_tags])

    with open(args.pred_file,'w',newline='') as csvfile:
        writer=csv.writer(csvfile)
        writer.writerows(prediction)

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
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

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