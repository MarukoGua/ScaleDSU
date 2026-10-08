#!/bin/bash 

config_file=./configs/xmega.json
save_dir=xmega

# python ./src/train_diffusion_sca.py --data xmega \
#     --data_file  ../data/XMEGA \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --step 200 \
#     --output_dir ./result/${save_dir}

# python ./src/train_dsu.py --data xmega \
#     --data_file  ../data/XMEGA \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --uncertainty 1 \
#     --factor 6 \
#     --output_dir ./result/$save_dir

factors='1 2 3 4 5 6 7 8 9'

for f in $factors
do
python ./src/train_ablation.py --data xmega \
    --data_file ../data/XMEGA \
    --source 01 \
    --targets 01,02,03,04,05,06,07,08 \
    --preprocess standardize \
    --config_file $config_file \
    --perturbation dsu \
    --uncertainty 1 \
    --factor $f \
    --noise_level 0 \
    --dropout 0 \
    --dense_type moe \
    --augment_position input \
    --output_dir ./result/$save_dir
done

# for f in $factors
# do
# python ./src/train_ablation.py --data xmega \
#     --data_file ../data/XMEGA \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --perturbation gaussian \
#     --uncertainty 0 \
#     --factor 1 \
#     --noise_level $f \
#     --dense_type mlp \
#     --augment_position internal \
#     --output_dir ./result/$save_dir
# done


# source=$1
# target=$2

# python train_mdm.py --data xmega \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#     --sources $source \
#     --target $target \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-only \
#     --factor 1 \
#     --uncertainty 0 \
#     --learning_rate 3e-4 \
#     --num_epoch 100 \
#     --use_scheduler False \
#     --output_dir ./experiments_dsu/xmega \
#     --save_prefix tl \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/xmega/trained_models/pretrain_tl.pt

# python train_alpa_baseline.py --data xmega \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#     --source 01 \
#     --targets 02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --learning_rate 3e-4 \
#     --use_scheduler False \
#     --num_epoch 100 \
#     --num_epoch_finetune 200 \
#     --learning_rate_finetune 0.001 \
#     --output_dir ./experiments_dsu/xmega \
#     --save_prefix alpa \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/xmega/trained_models/pretrain_tl.pt

# python train_cdpa_baseline.py --data xmega \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#     --source 01 \
#     --targets 02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --learning_rate 3e-4 \
#     --use_scheduler False \
#     --num_epoch 100 \
#     --nu2m_epoch_finetune 15 \
#     --learning_rate_finetune 0.001 \
#     --output_dir ./experiments_dsu/xmega \
#     --save_prefix cdpa \
#     --num_trace_attack 1000 \
#     --num_attack 100 \
#     --pretrain ./experiments_dsu/xmega/trained_models/pretrain_tl.pt

# python train_dsu.py --data xmega \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --uncertainty 1 \
#     --learning_rate 3e-4 \
#     --num_epoch 100 \
#     --use_scheduler False \
#     --random_search True \
#     --num_trials 50 \
#     --output_dir ./experiments_dsu/xmega/search \
#     --save_prefix dsu_search \
#     --num_trace_attack 1000 \
#     --num_attack 100


# factors="6"
# for f in $factors
# do
#     python train_dsu.py --data xmega \
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#         --source 01 \
#         --targets 01,02,03,04,05,06,07,08 \
#         --preprocess standardize \
#         --config_file $config_file \
#         --mode dsu-only \
#         --factor $f \
#         --uncertainty 1 \
#         --learning_rate 3e-4 \
#         --num_epoch 100 \
#         --use_scheduler False \
#         --output_dir ./experiments_dsu/xmega/source=01 \
#         --save_prefix ablation_k=$f \
#         --num_trace_attack 1000 \
#         --num_attack 100

# done




# python test_dsu.py --data xmega \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/XMEGA \
#     --source 01 \
#     --targets 01,02,03,04,05,06,07,08 \
#     --preprocess standardize \
#     --model_file ./experiments_dsu/xmega/trained_models/pretrain_tl.pt \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --output_dir ./experiments_dsu/xmega \
#     --save_prefix base