#!/bin/bash

# This enforces single head attention at multiple resolutions.
python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 64 \
    --multi_res \
    --fixed_heads 4 \
    --image_size 128 \
    --upsample_type resnet \
    --downsample_type resnet \
    --num_epochs 500 \
    --train_batch_size 20 \
    --eval_batch_size 20 \
    --wandb_run_name task1_ddpm_final_ablation_otter \
    --calculate_fid \
    --calculate_is \
    --verbose
