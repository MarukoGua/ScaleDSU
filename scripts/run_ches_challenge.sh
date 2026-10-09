#!/bin/bash 
config_file=./configs/ches_challenge.json
save_dir=ches_challenge

python ./src/train_dsu.py --data ches_challenge \
    --data_file  ../data/CHES_Challenge \
    --targets 0,1,2,3 \
    --preprocess standardize \
    --config_file $config_file \
    --mode dsu-scale \
    --uncertainty 1 \
    --factor 2 \
    --output_dir ./result/${save_dir} \
    --num_trace_attack -1 \
    --num_attack 100
