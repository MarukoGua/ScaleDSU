config_file=./configs/aespt_desyn.json
save_dir=aespt_desyn
implementation=MS1

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
