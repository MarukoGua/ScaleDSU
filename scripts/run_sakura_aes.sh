#!/bin/bash 

config_file=./configs/sakura.json
data=sakura_aes
data_file=../data/SAKURA_AES
source=1
targets=1,2,3
output_dir=./result/$data
num_trace_attack=5000
num_attack=100

python ./src/train_dsu.py --data $data \
    --data_file $data_file  \
    --source $source \
    --targets $targets \
    --preprocess standardize \
    --config_file $config_file \
    --mode dsu-scale \
    --uncertainty 1 \
    --factor 6 \
    --output_dir $output_dir \
    --num_trace_attack $num_trace_attack \
    --num_attack $num_attack \
