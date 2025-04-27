#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_ablation_ddim \
    --calculate_fid \
    --calculate_is \
    --verbose
