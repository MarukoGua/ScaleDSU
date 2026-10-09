#!/bin/bash 

config_file=./configs/xmega_em.json
save_dir=xmega_em

python ./src/train_dsu.py --data xmega_em \
    --data_file  ../data/XMEGA_EM \
    --source 01 \
    --targets 01,02,03,04,05,06,07,08 \
    --preprocess standardize \
    --config_file $config_file \
    --mode dsu-scale \
    --factor 3 \
    --uncertainty 1 \
    --output_dir ./result/$save_dir \
