# Sample Code for Homework 1 ADL NTU

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl-hw1"
make
conda activate adl-hw1
pip install -r requirements.txt
# Otherwise
pip instsall -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detectiona and slot tagging datasets
bash preprocess.sh
```

## Intent detection
```shell
python3 train_intent.py --device=cuda --dropout=0.2 --max_len=10 --num_epoch=50 --num_layer=2
```

## Slot Tagging
```shell
python3 train_slot.py --device=cuda --max_len=20 --weight_decay=1e-5 --num_layer=3 --num_epoch=50
```

## Plot Figures for the Report
```shell
python3 train_slot.py --device=cuda --max_len=20 --weight_decay=1e-5 --num_layer=3 --num_epoch=50 --plot_figure=True
```

## Download the Models
```shell
bash download.sh
```
