#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 2000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name steps_2000 \
    --calculate_fid \
    --calculate_is \
    --verbose
