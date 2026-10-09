#!/bin/bash 

config_file=./configs/aespt.json
save_dir=aespt
implementation=MS1

python ./src/train_dsu.py --data aespt \
    --data_file ../data/AES_PTv2 \
    --source D1 \
    --targets D1,D2,D3,D4 \
    --implementation $implementation \
    --preprocess no_preprocess \
    --config_file $config_file \
    --mode dsu-scale \
    --uncertainty 1 \
    --factor 7 \
    --output_dir ./result/${save_dir} \
    --num_trace_attack 1000 \
    --num_attack 100
