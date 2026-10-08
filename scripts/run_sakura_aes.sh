#!/bin/bash 

config_file=./configs/sakura.json
data=sakura_aes
data_file=../data/SAKURA_AES
source=1
targets=1,2,3
output_dir=./result/$data
num_trace_attack=5000
num_attack=100

# python ./src/train_diffusion_sca.py --data $data \
#     --data_file  $data_file \
#     --source $source \
#     --targets $targets \
#     --preprocess standardize \
#     --config_file $config_file \
#     --step 200 \
#     --output_dir $output_dir \
#     --num_trace_attack $num_trace_attack \
#     --num_attack $num_attack

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


# python train_dsu.py --data $data \
#     --data_file $data_file  \
#     --source $source \
#     --targets $targets \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --uncertainty 1 \
#     --factor 5 \
#     --learning_rate ${learning_rate} \
#     --use_scheduler $use_scheduler \
#     --num_epoch $num_epoch \
#     --random_search True \
#     --num_trials 100 \
#     --output_dir $output_dir \
#     --save_prefix dsu_search \
#     --num_trace_attack $num_trace_attack \
#     --num_attack $num_attack \


# python train_mdm.py --data $data \
#     --data_file  $data_file \
#     --sources 1,2 \
#     --target 3 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --learning_rate ${learning_rate} \
#     --num_epoch $num_epoch \
#     --use_scheduler $use_scheduler \
#     --output_dir $output_dir \
#     --save_prefix mdm_source=1,2 \
#     --num_trace_attack $num_trace_attack \
#     --num_attack $num_attack \
#     --pretrain $output_dir/trained_models/base.pt

# python train_mdm.py --data $data \
#     --data_file  $data_file \
#     --sources 1,3 \
#     --target 2 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --learning_rate ${learning_rate} \
#     --num_epoch $num_epoch \
#     --use_scheduler $use_scheduler \
#     --output_dir $output_dir \
#     --save_prefix mdm_source=1,3 \
#     --num_trace_attack $num_trace_attack \
#     --num_attack $num_attack \
#     --pretrain $output_dir/trained_models/base.pt


# python train_alpa_baseline.py --data $data \
#     --data_file  $data_file \
#     --source $source \
#     --targets 2,3 \
#     --config_file $config_file \
#     --preprocess standardize \
#     --learning_rate $learning_rate \
#     --use_scheduler $use_scheduler \
#     --num_epoch $num_epoch \
#     --num_epoch_finetune 30 \
#     --learning_rate_finetune 0.0004 \
#     --output_dir $output_dir \
#     --save_prefix alpa \
#     --num_trace_attack 5000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/sakura_aes/trained_models/alpa-pretrain.pt

# python train_cdpa_baseline.py --data $data \
#     --data_file  $data_file \
#     --source $source \
#     --targets 2,3 \
#     --config_file $config_file \
#     --preprocess standardize \
#     --learning_rate $learning_rate \
#     --use_scheduler $use_scheduler \
#     --num_epoch $num_epoch \
#     --num_epoch_finetune 30 \
#     --learning_rate_finetune 0.001 \
#     --output_dir $output_dir \
#     --save_prefix cdpa \
#     --num_trace_attack 5000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/sakura_aes/trained_models/alpa-pretrain.pt