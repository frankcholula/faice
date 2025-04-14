#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_ddpm_linear_NoAug \
    --calculate_fid \
    --calculate_is
#    --gblur
#    --RHFlip


#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --model unet_resnet \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_resnet512_ddpm_linear \
#    --calculate_fid \
#    --calculate_is

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 5 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_ddpm_linear_test \
    --calculate_fid \
    --calculate_is \
    --no_wandb