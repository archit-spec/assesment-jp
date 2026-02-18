#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python3 train_sft_trackb.py \
    --model_name Qwen/Qwen2.5-Coder-1.5B \
    --data_path data/sft_trackb_train.json \
    --output_dir results/track_b_sft \
    --max_seq_length 1024 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4
