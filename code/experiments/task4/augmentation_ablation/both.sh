#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --gblur \
    --RHFlip \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --wandb_run_name task4_ablation_rhflip_gblur_v2 \
    --calculate_fid \
    --calculate_is \
    --verbose