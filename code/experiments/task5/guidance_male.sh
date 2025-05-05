#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --pipeline cond \
    --beta_schedule linear \
    --model unet \
    --unet_variant cond \
    --condition_on male \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name task5_baseline \
    --calculate_fid \
    --calculate_is \
    --verbose
