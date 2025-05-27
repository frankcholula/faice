#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 2000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name adm_increase_steps \
    --wandb_run_name task4_bs16_baseline_steps_2000 \
    --calculate_fid \
    --calculate_is \
    --verbose
