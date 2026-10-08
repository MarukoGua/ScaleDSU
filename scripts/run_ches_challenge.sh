#!/bin/bash 
config_file=./configs/ches_challenge.json
save_dir=ches_challenge

python ./src/train_diffusion_sca.py --data ches_challenge \
    --data_file  ../data/CHES_Challenge \
    --targets 0,1,2,3 \
    --preprocess standardize \
    --config_file $config_file \
    --step 200 \
    --output_dir ./result/${save_dir} \
    --num_trace_attack -1 \
    --num_attack 100

# python ./src/train_dsu.py --data ches_challenge \
#     --data_file  ../data/CHES_Challenge \
#     --targets 0,1,2,3 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --uncertainty 1 \
#     --factor 2 \
#     --output_dir ./result/${save_dir} \
#     --num_trace_attack -1 \
#     --num_attack 100

# python train_dsu.py --data ches_challenge \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#     --targets 0,1,2,3 \
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --uncertainty 1 \
#     --learning_rate 0.0008103063234970507 \
#     --num_epoch 190 \
#     --use_scheduler True \
#     --random_search True \
#     --num_trials 100 \
#     --output_dir ./experiments_dsu/${save_dir}/search \
#     --save_prefix dsu_search \
#     --num_trace_attack -1 \
#     --num_attack 100

# python test_dsu.py --data ches_challenge \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#     --targets 0,1,2,3\
#     --preprocess standardize \
#     --config_file $config_file \
#     --mode dsu-only \
#     --uncertainty 0 \
#     --model_file ./experiments_dsu/${save_dir}/trained_models/base.pt \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix base \
#     --num_trace_attack -1 \
#     --num_attack 100

# pretrain=/home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/experiments_dsu/${save_dir}/trained_models/cdpa-pretrain.pt

# targets="0 1 2 3"

# for t in $targets
# do
#     python train_cdpa_baseline.py --data ches_challenge \
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#         --targets $t \
#         --preprocess standardize_horizontal \
#         --config_file $config_file \
#         --learning_rate 0.0008103063234970507 \
#         --use_scheduler True \
#         --num_epoch 190 \
#         --num_epoch_finetune 15 \
#         --learning_rate_finetune 0.001 \
#         --output_dir ./experiments_dsu/${save_dir} \
#         --save_prefix cdpa \
#         --num_trace_attack -1 \
#         --num_attack 100 \
#         --pretrain $pretrain
# done

# for t in $targets
# do
#     python train_alpa_baseline.py --data ches_challenge\
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#         --config_file $config_file \
#         --targets $t \
#         --learning_rate 0.0008103063234970507 \
#         --use_scheduler True \
#         --num_epoch 190 \
#         --num_epoch_finetune 15 \
#         --learning_rate_finetune 0.001 \
#         --output_dir ./experiments_dsu/${save_dir} \
#         --save_prefix alpa \
#         --num_trace_attack -1 \
#         --num_attack 100 \
#         --pretrain $pretrain
# done
# sources="0 1 2 3"
# for s in $sources
# do
#     t=$(( (10#$s + 1) % 4 ))
#     python train_mdm.py --data ches_challenge \
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#         --sources $s \
#         --target $t \
#         --preprocess standardize \
#         --config_file $config_file \
#         --mode dsu-only \
#         --factor 1 \
#         --uncertainty 0 \
#         --learning_rate 0.0008103063234970507 \
#         --num_epoch 190 \
#         --use_scheduler True \
#         --output_dir ./experiments_dsu/${save_dir} \
#         --save_prefix tl_$t \
#         --num_trace_attack -1 \
#         --num_attack 100 \
#         --pretrain ./experiments_dsu/${save_dir}/trained_models/pretrain_tl.pt \
#         --finetune ./experiments_dsu/${save_dir}/trained_models/ft_tl_$t.pt
# done


# python train_diffusion_sca.py --data ches_challenge \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/CHES_Challenge \
#     --preprocess standardize \
#     --config_file $config_file \
#     --learning_rate 0.0008103063234970507 \
#     --use_scheduler True \
#     --num_epoch 190 \
#     --step 200 \
#     --output_dir ./experiments_dsu/${save_dir} \
#     --save_prefix diffusion \
#     --num_trace_attack -1 \
#     --num_attack 100