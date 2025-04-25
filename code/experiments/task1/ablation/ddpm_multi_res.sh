#!/bin/bash

# This enforces single head attention at multiple resolutions. (32x32, 16x16, 8x8)

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 64 \
    --multi_res \
    --upsample_type conv \
    --downsample_type conv \
    --fixed_heads 1 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 20 \
    --eval_batch_size 20 \
    --wandb_run_name task1_ddpm_multi_res_ablation \
    --calculate_fid \
    --calculate_is \
    --verbose
