#!/bin/bash

# This increases the number of attention heads to 4 by decreasing the head dimension from 128 to 64.


python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name task1_ddpm_head_dim_64_ablation \
    --calculate_fid \
    --calculate_is \
    --verbose
