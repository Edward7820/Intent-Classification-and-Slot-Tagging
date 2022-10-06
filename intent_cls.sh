# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_intent.py --test_file data/intent/test.json --ckpt_path ckpt/intent/model.pth --device=cuda --dropout=0.2 --max_len=10
