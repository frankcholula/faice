#!/bin/bash

# This is baseline for DDPM. 128 channels, a depth of 2, 1 attention head, and no multi-res attention.
# Run this first.
python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 256 \
    --upsample_type conv \
    --downsample_type conv \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 20 \
    --eval_batch_size 20 \
    --wandb_run_name task1_ddpm_base \
    --calculate_fid \
    --calculate_is
    --verbose