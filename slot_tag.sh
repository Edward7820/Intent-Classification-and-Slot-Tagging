# "${1}" is the first argument passed to the script
# "${2}" is the second argument passed to the script
python3 test_slot.py --test_file="${1}" --pred_file="${2}" --ckpt_path="ckpt/slot/model.pth" --device=cuda --max_len=20 --num_layer=3
