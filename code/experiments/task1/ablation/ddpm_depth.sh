#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --layers_per_block 4 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 14 \
    --eval_batch_size 14 \
    --wandb_run_name task1_ddpm_depth_ablation_otter \
    --calculate_fid \
    --calculate_is \
    --verbose
