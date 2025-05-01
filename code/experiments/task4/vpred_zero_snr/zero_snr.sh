#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --beta_schedule squaredcos_cap_v2 \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --rescale_betas_zero_snr \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_ablation_zero_snr \
    --calculate_fid \
    --calculate_is \
    --verbose
