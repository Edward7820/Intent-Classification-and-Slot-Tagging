import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    def collate_fn1(samples: List[Dict]) -> Dict:
        return datasets[TRAIN].collate_fn(samples)
    def collate_fn2(samples: List[Dict]) -> Dict:
        return datasets[DEV].collate_fn(samples)
    collate_fns={TRAIN: collate_fn1, DEV: collate_fn2}
    dataloaders: Dict[str, DataLoader] = {
        split: DataLoader(datasets[split],batch_size=args.batch_size, 
        collate_fn=collate_fns[split], shuffle=True, pin_memory=True)
        for split in SPLITS
    }
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    device = torch.device(args.device)
    model = SeqTagger(embeddings=embeddings,hidden_size=args.hidden_size, 
    num_layers=args.num_layers,dropout=args.dropout,
    bidirectional=args.bidirectional, num_class=datasets[TRAIN].num_classes,
    device=device)
    model = model.to(device)

    # print(datasets[TRAIN].ignore_idx)
    criterion = torch.nn.CrossEntropyLoss(ignore_index = datasets[TRAIN].ignore_idx,reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), 
    lr = args.lr, weight_decay = args.weight_decay)

    loss_data = {'training loss':list([]), 'evaluation loss': list([])}
    accuracy_data = {'training accuracy':([]), 'evaluation accuracy':([])}
    best_accuracy = 0
    best_loss = 10000
    model_saved_epoch = 0
    model_path = args.ckpt_dir / "model.pth"
    train_token_nums = datasets[TRAIN].token_nums
    eval_token_nums = datasets[DEV].token_nums
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # Training loop
        correct = 0
        loss_sum=0
        train_batch_num=0
        train_data_size = 0
        for train_data in dataloaders[TRAIN]:
            model.train()
            train_data['tokens']=train_data['tokens'].to(device)
            train_data['tags']=train_data['tags'].to(device)
            output = (model(train_data['tokens']))['prediction']
            output = output.transpose(1, 2)
            loss = criterion(output, train_data['tags'])
            loss_sum += loss.item()

            train_batch_num+=1
            train_batch_size = output.size()[0]
            train_data_size += train_batch_size
            seq_num = output.size()[2]
            for i in range(train_batch_size):
                if train_token_nums[train_data['id'][i]] > seq_num:
                    continue
                all_correct = True
                for j in range(seq_num):
                    if train_data['tags'][i][j].item()!=datasets[TRAIN].ignore_idx and \
                    torch.argmax(output[i,:,j]).item()!=train_data['tags'][i][j].item():
                        all_correct = False
                        break
                if all_correct:
                    correct += 1
            
            optimizer.zero_grad()
            loss.backward()
            if args.do_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip,
                error_if_nonfinite=True)
            optimizer.step()
            # print('    Batch: {}/117.............'.format(batch_num), end=' ')
            # print("    Loss: {:.4f}".format(loss.item()))
        print('Epoch: {}/{}.............'.format(epoch,args.num_epoch), end=' ')
        print("Train loss: {:.5f}".format(loss_sum/train_batch_num),end=' ')
        print("Train accuracy: {}/{}".format(correct,train_data_size))
        loss_data['training loss'].append(loss_sum/train_batch_num)
        accuracy_data['training accuracy'].append(correct/train_data_size)

        #evaluation loop
        correct=0
        eval_data_size=0
        loss_sum=0
        eval_batch_num=len(dataloaders[DEV])
        for eval_data in dataloaders[DEV]:
            model.eval()
            eval_data['tokens']=eval_data['tokens'].to(device)
            eval_data['tags']=eval_data['tags'].to(device)
            output = (model(eval_data['tokens']))['prediction']
            output = output.transpose(1, 2)
            # output shape: batch_size * num_classes * max_len
            loss = criterion(output, eval_data['tags'])
            eval_batch_size = output.size()[0]
            seq_num = output.size()[2]
            eval_data_size += eval_batch_size
            loss_sum += loss.item()
            # print(output.size())
            for i in range(eval_batch_size):
                if eval_token_nums[eval_data['id'][i]] > seq_num:
                    continue
                all_correct = True
                for j in range(seq_num):
                    if eval_data['tags'][i,j].item()!=datasets[DEV].ignore_idx and \
                    torch.argmax(output[i,:,j]).item()!=eval_data['tags'][i,j].item():
                        # if epoch%10==3:
                        #    print(j, torch.argmax(output[i][:][j]).item(), eval_data['tags'][i][j].item())
                        all_correct = False
                        break
                if all_correct:
                    correct += 1
        accuracy = correct/eval_data_size
        eval_loss=loss_sum/eval_batch_num
        if (eval_loss < best_loss):
            best_loss = eval_loss
        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            if accuracy >= 0.72:
                torch.save(model.state_dict(),model_path)
                model_saved_epoch = epoch
                print("model is saved")
        print('Epoch: {}/{}.............'.format(epoch,args.num_epoch), end=' ')
        print("Loss: {:.5f}".format(eval_loss), end=' ')
        print("Accuracy: {}/{}".format(correct,eval_data_size))
        loss_data['evaluation loss'].append(eval_loss)
        accuracy_data['evaluation accuracy'].append(accuracy)
        if epoch%10==9:
            print("Max accuracy: {:.5f} Min loss: {:.5f} Model saved at epoch {}".format(best_accuracy,
            best_loss,model_saved_epoch))
    
    loss_df = pd.DataFrame(data=loss_data,dtype=float)
    accuracy_df = pd.DataFrame(data=accuracy_data,dtype=float)
    loss_df.plot()
    accuracy_df.plot()
    plt.show()

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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

# python3 train_slot.py --device=cuda --max_len=20 --weight_decay=1e-5
# GRU
# max_len:20 (default setting)
# Max accuracy: 0.77900 Min loss: 0.12243

# 1
# max_len: 20
# dropout: 0.15
# Max accuracy: 0.77300 Min loss: 0.12226

# 2
# max_len: 20
# dropout: 0.1
# do_grad_clip: True

# 3
# max_len = 20
# hidden_size = 600
# Max accuracy: 0.77500 Min loss: 0.12188

# 4
# max_len = 20
# hidden_size = 400
# Max accuracy: 0.77900 Min loss: 0.12053

# LSTM
# 5
# max_len = 20 (default setting)
# Max accuracy: 0.78500 Min loss: 0.11988

# 6
# max_len = 20
# weight decay = 1e-5