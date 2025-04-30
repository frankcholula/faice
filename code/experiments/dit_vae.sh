#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model dit_transformer \
    --pipeline dit_vae \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 20 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 40 \
    --eval_batch_size 40 \
    --wandb_run_name liang_dit_vae_ddpm_linear_batch_size_40_test \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb

