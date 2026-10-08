config_file=./configs/aespt_desyn.json
save_dir=aespt_desyn
implementation=MS1


# python ./src/train_diffusion_sca.py --data aespt \
#     --data_file ../data/AES_PTv2 \
#     --source D1 \
#     --targets D1,D2,D3,D4 \
#     --preprocess shift \
#     --implementation $implementation \
#     --config_file $config_file \
#     --step 200 \
#     --output_dir ./result/${save_dir} \
#     --num_trace_attack 1000 \
#     --num_attack 100

python ./src/train_dsu.py --data aespt \
    --data_file ../data/AES_PTv2 \
    --source D1 \
    --targets D1,D2,D3,D4 \
    --implementation $implementation \
    --preprocess shift \
    --config_file $config_file \
    --mode dsu-scale \
    --factor 1 \
    --uncertainty 1 \
    --output_dir ./result/${save_dir} \
    --num_trace_attack 1000 \
    --num_attack 100

# factors="9 10"
# for f in $factors
# do

# python train_dsu.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source $source \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --preprocess shift \
#     --config_file $config_file \
#     --mode dsu-scale \
#     --learning_rate 9e-4 \
#     --factor $f \
#     --uncertainty 1 \
#     --use_scheduler True \
#     --num_epoch 150 \
#     --output_dir ./experiments_dsu/${save_dir}/source=${source}_${implementation} \
#     --save_prefix ablation_k=$f \
#     --num_trace_attack 1000 \
#     --num_attack 100
# done

# python train_optuna.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source $source \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --preprocess shift \
#     --batch_size 512 \
#     --learning_rate 9e-4 \
#     --use_scheduler False \
#     --output_dir ./experiments/${save_dir}/source=${source}_${implementation} \
#     --save_prefix optuna \
#     --study_name aespt_${source}_${implementation}_desync \

# sources="D4"
# for source in $sources
# do
#     python train_alpa_baseline.py --data aespt\
#             --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#             --source $source \
#             --targets D1,D2,D3,D4 \
#             --implementation $implementation \
#             --preprocess shift \
#             --config_file $config_file \
#             --learning_rate 0.0012 \
#             --use_scheduler False \
#             --num_epoch 70 \
#             --num_epoch_finetune 50 \
#             --learning_rate_finetune 0.001 \
#             --output_dir ./experiments/${save_dir}/source=${source}_${implementation} \
#             --save_prefix alpa \
#             --num_trace_attack 1000 \
#             --num_attack 100 \

#     pretrain=/home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/experiments/${save_dir}/source=${source}_${implementation}/trained_models/alpa-pretrain.pt
#     python train_cdpa_baseline.py --data aespt \
#         --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#         --source $source \
#         --targets D1,D2,D3,D4 \
#         --implementation $implementation \
#         --preprocess shift \
#         --config_file $config_file \
#         --learning_rate 0.0012 \
#         --use_scheduler False \
#         --num_epoch 80 \
#         --num_epoch_finetune 50 \
#         --learning_rate_finetune 0.001 \
#         --output_dir ./experiments/${save_dir}/source=${source}_${implementation} \
#         --save_prefix cdpa \
#         --num_trace_attack 1000 \
#         --num_attack 100 \
#         --pretrain $pretrain
# done

# steps='100 200 300 400 500 600 700 800 900 1000'
# for step in $steps
# do
# python train_diffusion_sca.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source $source \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --preprocess shift \
#     --config_file $config_file \
#     --learning_rate 0.004 \
#     --batch_size 512 \
#     --use_scheduler True \
#     --num_epoch 150 \
#     --step 200 \
#     --output_dir ./experiments_dsu/${save_dir}/source=${source}_${implementation} \
#     --save_prefix diffusion \
#     --num_trace_attack 1000 \
#     --num_attack 100

# python train_mdm.py --data aespt \
#     --data_file  /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --sources D1,D4 \
#     --target D2 \
#     --implementation $implementation \
#     --preprocess shift \
#     --uncertainty 0 \
#     --mode dsu-only \
#     --config_file $config_file \
#     --learning_rate 1e-3 \
#     --use_scheduler True \
#     --num_epoch 150 \
#     --output_dir ./experiments_dsu/${save_dir}/mdm \
#     --save_prefix mdm \
#     --num_trace_attack 1000 \
#     --num_attack 100


# done
# python test_diffusion.py --data aespt \
#     --data_file /home/chenyimeng/workspace/SideChannelAnalysis/data/AES_PTv2 \
#     --source $source \
#     --targets D1,D2,D3,D4 \
#     --implementation $implementation \
#     --preprocess shift \
#     --config_file $config_file \
#     --model_file /home/chenyimeng/workspace/SideChannelAnalysis/sca-portability/experiments/aespt_desync/source=D1_MS1/trained_models/diffusion.pt \
#     --output_dir ./experiments/${save_dir}/source=${source}_${implementation} \
#     --save_prefix diffusion