import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # create data loader
    def collate_fn1(samples: List[Dict]) -> Dict:
        return datasets[TRAIN].collate_fn(samples)
    def collate_fn2(samples: List[Dict]) -> Dict:
        return datasets[DEV].collate_fn(samples)
    collate_fns=dict()
    collate_fns[TRAIN]=collate_fn1
    collate_fns[DEV]=collate_fn2
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(datasets[split],batch_size=args.batch_size, collate_fn=collate_fns[split], pin_memory=True)
        for split in SPLITS
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    is_cuda = torch.cuda.is_available()
    if (is_cuda==0):
        device = torch.device("cpu")
        print("cuda not available")
    else:
        device = torch.device(args.device)
    model = Model(embeddings=embeddings,hidden_size=args.hidden_size,
    num_layers=args.num_layers,dropout=args.dropout,bidirectional=args.bidirectional,
    num_class=datasets[TRAIN].num_classes())
    model.to(device)

    # TODO: init optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters,lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        for i, (inputs, labels) in enumerate(dataloaders[TRAIN]):

        

    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
