# Intent Classification and Slot Tagging

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "nlp-env"
make
conda activate nlp-env
pip install -r requirements.txt
# Otherwise
pip instsall -r requirements.in
```

## Preprocessing
```shell
# To preprocess intent detection and slot tagging datasets
bash preprocess.sh
```

## Training data format
https://github.com/ntu-adl-ta/ADL21-HW1/tree/main/data

## Training for intent detection
```shell
python3 train_intent.py --device=cuda --dropout=0.2 --max_len=10 --num_epoch=50 --num_layer=2
```

## Training for slot tagging
```shell
python3 train_slot.py --device=cuda --max_len=20 --weight_decay=1e-5 --num_layer=3 --num_epoch=50
```

## Plot training figures
```shell
python3 train_slot.py --device=cuda --max_len=20 --weight_decay=1e-5 --num_layer=3 --num_epoch=50 --plot_figure=True
```

## Download the models
```shell
bash download.sh
```

## Prediction
```shell
bash ./intent_cls.sh /path/to/test.json /path/to/pred.csv
bash ./slot_tag.sh /path/to/test.json /path/to/pred.csv
```

## Seqeval evaluation
```shell
bash seqeval.sh /path/to/test.json
```

## Reference
https://github.com/ntu-adl-ta/ADL21-HW1