#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --rescale_betas_zero_snr \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name task4_ablation_ddpm_0snr_v3 \
    --calculate_fid \
    --calculate_is \
    --verbose
