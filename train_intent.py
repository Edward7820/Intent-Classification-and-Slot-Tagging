import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

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
        split: DataLoader(datasets[split],batch_size=args.batch_size, 
        collate_fn=collate_fns[split], shuffle=True, pin_memory=True)
        for split in SPLITS
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    device = torch.device(args.device)
    model = SeqClassifier(embeddings=embeddings,hidden_size=args.hidden_size, 
    num_layers=args.num_layers,dropout=args.dropout,
    bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes,
    device=device)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
    lr = args.lr, weight_decay = args.weight_decay)

    loss_data = {'training loss':list([]), 'evaluation loss': list([])}
    accuracy_data = {'evaluation accuracy':list([])}
    best_accuracy = 0
    best_loss = 1.5
    model_path = args.ckpt_dir / "model.pth"
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        # TODO: Evaluation loop - calculate accuracy and save model weights
        loss_sum=0
        train_batch_num=0
        for batch_num, train_data in enumerate(dataloaders[TRAIN]):
            model.train()
            train_batch_num+=1
            train_data['text']=train_data['text'].to(device)
            train_data['intent']=train_data['intent'].to(device)
            output = (model(train_data['text']))['prediction']
            loss = criterion(output, train_data['intent'])
            loss_sum+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            if args.do_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip,
                error_if_nonfinite=True)
            optimizer.step()
        print('Epoch: {}/{}.............'.format(epoch,args.num_epoch), end=' ')
        print("Train loss: {:.5f}".format(loss_sum/train_batch_num))
        loss_data['training loss'].append(loss_sum/train_batch_num)

        correct=0
        eval_data_size=0
        loss_sum=0
        eval_batch_num=len(dataloaders[DEV])
        for batch_num, eval_data in enumerate(dataloaders[DEV]):
            model.eval()
            eval_data['text']=eval_data['text'].to(device)
            eval_data['intent']=eval_data['intent'].to(device)
            output = (model(eval_data['text']))['prediction']
            loss = criterion(output, eval_data['intent'])
            eval_batch_size = eval_data['intent'].size()[0]
            eval_data_size += eval_batch_size
            loss_sum += loss.item()
            for i in range(eval_batch_size):
                if torch.argmax(output[i]).item()==eval_data['intent'][i].item():
                    correct+=1
        accuracy = correct/eval_data_size
        eval_loss=loss_sum/eval_batch_num
        if (eval_loss < best_loss):
            best_loss = eval_loss
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            if accuracy >= 0.88:
                torch.save(model.state_dict(),model_path)
                print("model is saved")
        print('Epoch: {}/{}.............'.format(epoch,args.num_epoch), end=' ')
        print("Loss: {:.5f}".format(eval_loss), end=' ')
        print("Accuracy: {}/{}".format(correct,eval_data_size))
        if epoch%10==9:
            print("Max accuracy: {:.5f} Min loss: {:.5f}".format(best_accuracy,best_loss))
        loss_data['evaluation loss'].append(eval_loss)
        accuracy_data['evaluation accuracy'].append(accuracy)

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
    parser.add_argument("--max_len", type=int, default=8)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--do_grad_clip", type=bool, default=False)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--weight_decay",type=float,default=0)

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