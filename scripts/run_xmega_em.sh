#!/bin/bash 

config_file=./configs/xmega_em.json
save_dir=xmega_em

# python ./src/train_diffusion_sca.py --data xmega_em \
#     --data_file  ../data/XMEGA_EM \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --step 200 \
#     --output_dir ./result/${save_dir}


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

# sources=$1
# target=$2
# python train_mdm.py --data xmega_em \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
#     --sources $sources \
#     --target $target \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-only \
#     --factor 1 \
#     --uncertainty 0 \
#     --learning_rate 3e-4 \
#     --num_epoch 200 \
#     --use_scheduler False \
#     --output_dir ./experiments_dsu/$save_dir \
#     --save_prefix tl \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/xmega_em/trained_models/pretrain_tl.pt

# pretrain=/home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/experiments_dsu/${save_dir}/trained_models/pretrain_tl.pt
# python test_dsu.py --data xmega_em \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --model_file $pretrain \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --output_dir ./experiments_dsu/xmega_em \
#     --save_prefix base

# python train_alpa_baseline.py --data xmega_em\
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
#     --source 01 \
#     --targets 02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --learning_rate 3e-4 \
#     --use_scheduler False \
#     --num_epoch 200 \
#     --num_epoch_finetune 200 \
#     --learning_rate_finetune 0.001 \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix alpa \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain $pretrain

# python train_cdpa_baseline.py --data xmega_em \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
#     --source 01 \
#     --targets 02,03,04,05,06,07,08 \
#     --config_file $config_file \
#     --preprocess standardize \
#     --learning_rate 3e-4 \
#     --use_scheduler False \
#     --num_epoch 200 \
#     --num_epoch_finetune 200 \
#     --learning_rate_finetune 0.001 \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix cdpa \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain $pretrain

# python train_diffusion_sca.py --data xmega_em \
    # --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
    # --source 01 \
    # --targets 02,03,04,05,06,07,08 \
    # --config_file $config_file \
    # --learning_rate 3e-4 \
    # --use_scheduler False \
    # --num_epoch 200 \
    # --step 200 \
    # --output_dir ./experiments_dsu/${save_dir} \
    # --save_prefix diffusion \
    # --num_trace_attack 1000 \
    # --num_attack 100 \

# factors='1 2 3 4 5 6 7 8 9 10'
# for f in $factors
# do
#     python train_dsu.py --data xmega_em \
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA_EM \
#         --source 01 \
#         --targets 01,02,03,04,05,06,07,08 \
#         --preprocess standardize \
#         --config_file $config_file \
#         --mode dsu-scale \
#         --learning_rate 3e-4 \
#         --factor $f \
#         --uncertainty 1 \
#         --num_epoch 200 \
#         --use_scheduler False \
#         --output_dir ./experiments_dsu/xmega_em/source=01 \
#         --save_prefix ablation_k=$f \
#         --num_trace_attack 1000 \
#         --num_attack 100
# done
