#!/bin/bash

python main.py \
    --dataset face \
    --scheduler "ddpm" \
    --beta_schedule "squaredcos_cap_v2" \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --beta_schedule "squaredcos_cap_v2" \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --image_size 128 \
    --gblur \
    --wandb_run_name "emre_ddpm_squaredcos_gblur_test" \
    --calculate_fid \
    --calculate_is \
    --verbose \
    --no_confirm

python main.py \
    --dataset face \
    --scheduler "ddpm" \
    --beta_schedule "squaredcos_cap_v2" \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --beta_schedule "squaredcos_cap_v2" \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --image_size 128 \
    --RHFlip \
    --wandb_run_name "emre_ddpm_squaredcos_RHFlip_test" \
    --calculate_fid \
    --calculate_is \
    --verbose \
    --no_confirm

python main.py \
    --dataset face \
    --scheduler "ddpm" \
    --beta_schedule "squaredcos_cap_v2" \
    --num_epochs 10 \
    --learning_rate 2e-4 \
    --beta_schedule "squaredcos_cap_v2" \
    --train_batch_size 48 \
    --eval_batch_size 48 \
    --image_size 128 \
    --RHFlip \
    --gblur \
    --wandb_run_name "emre_ddpm_squaredcos_gblur_RHFlip_test" \
    --calculate_fid \
    --calculate_is \
    --verbose \
    --no_confirm