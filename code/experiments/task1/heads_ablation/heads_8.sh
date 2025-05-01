#!/bin/bash

# This increases the number of attention heads to 8 by decreasing the head dimension from 256 to 32.

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 32 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name heads_8 \
    --calculate_fid \
    --calculate_is \
    --verbose
