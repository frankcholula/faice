#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --eta 0.0 \
    --num_inference_steps 100 \
    --attention_head_dim 64 \
    --rescale_betas_zero_snr \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_ddim_linear_0snr \
    --calculate_fid \
    --calculate_is \
    --verbose

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_baseline \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
