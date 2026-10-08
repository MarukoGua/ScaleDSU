#!/bin/bash 

config_file=./configs/aespt.json
save_dir=aespt
implementation=MS1

# python ./src/train_diffusion_sca.py --data aespt \
#     --data_file ../data/AES_PTv2 \
#     --source D1 \
#     --targets D1,D2,D3,D4 \
#     --preprocess standardize \
#     --implementation $implementation \
#     --config_file $config_file \
#     --step 500 \
#     --output_dir ./result/${save_dir} \
#     --num_trace_attack 1000 \
#     --num_attack 100

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

# python train_mdm.py --data aespt \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --sources D1,D2 \
#     --target D3 \
#     --implementation $implementation \
#     --preprocess no_preprocess \
#     --uncertainty 0 \
#     --mode dsu-only \
#     --config_file $config_file \
#     --learning_rate 5e-4 \
#     --num_epoch 150 \
#     --use_scheduler False \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix tl \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/${save_dir}/trained_models/pretrain_tl.pt

# python test_dsu.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source D1 \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --preprocess no_preprocess \
#     --model_file ./experiments_dsu/${save_dir}/trained_models/pretrain_tl.pt \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix base



# factors="8"
# for f in $factors
# do

#     python train_dsu.py --data aespt \
#         --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#         --source $source \
#         --targets D1,D2,D3,D4 \
#         --implementation $implementation \
#         --preprocess no_preprocess \
#         --config_file $config_file \
#         --mode dsu-scale \
#         --learning_rate 6e-4 \
#         --factor $f \
#         --uncertainty 1 \
#         --use_scheduler False \
#         --num_epoch 150 \
#         --output_dir ./experiments_dsu/${save_dir}/source=${source}_${implementation} \
#         --save_prefix ablation_k=$f \
#         --num_trace_attack 1000 \
#         --num_attack 100
# done
# python train_optuna.py --data aespt \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PT_v2 \
#     --source $source \
#     --targets D2,D3,D4 \
#     --implementation $implementation \
#     --output_dir ./experiments/${save_dir}_${source}_${implementation} \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --study_name ${save_dir}_${source}_${implementation} 

# if [ "$implementation" = "Unprotected" ] ;then
#     lr=0.0044387897140694505
#     epoch=190
#     scheduler=True
# elif [ "$implementation" = "MS1" ] ;then
#     lr=0.0009785124662782443
#     epoch=180
#     scheduler=False
# fi

# source=D1
# implementation=Unprotected
# config_file=/home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/network_configs/aespt/aespt_D1_${implementation}.json
# # config_file=/home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/network_configs/xmega/config_xmega_01.json
# python train_diffusion_sca.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source $source \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --config_file $config_file \
#     --learning_rate 0.0007 \
#     --num_epoch 100 \
#     --step 500 \
#     --output_dir ./experiments_dsu/${save_dir}/source=${source}_${implementation} \
#     --save_prefix diffusion \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
