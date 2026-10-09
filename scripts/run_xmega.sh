#!/bin/bash 

config_file=./configs/xmega.json
save_dir=xmega

python ./src/train_dsu.py --data xmega \
    --data_file  ../data/XMEGA \
    --source 01 \
    --targets 01,02,03,04,05,06,07,08 \
    --preprocess standardize \
    --config_file $config_file \
    --mode dsu-scale \
    --uncertainty 1 \
    --factor 6 \
    --output_dir ./result/$save_dir