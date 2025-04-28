#!/bin/bash

# This enforce attention at 3 different resolutions.
# 1 head with 256 channels at 32x32, 1 head with 256 channels at 16x16, and 2 heads with 512 channels at 8x8.

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 256 \
    --multi_res \
    --image_size 128 \
    --upsample_type conv \
    --downsample_type conv \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name task1_ddpm_multi_res_ablation \
    --calculate_fid \
    --calculate_is \
    --verbose
