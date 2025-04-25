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
    --num_epochs 500 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name task1_ddpm_final_ablation_otter \
    --calculate_fid \
    --calculate_is \
    --verbose
