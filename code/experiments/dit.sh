#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model dit_transformer \
    --pipeline dit \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 50 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name liang_dit_ddpm_linear \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb
