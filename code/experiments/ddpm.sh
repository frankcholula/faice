#!/bin/bash

#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_ddpm_linear_NoAug \
#    --calculate_fid \
#    --calculate_is
#    --gblur
#    --RHFlip

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_unet_ddpm_linear_attention_head_dim_64 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb


#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --model unet_resnet512 \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_resnet512_ddpm_linear_3attention_head_dim8 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm

#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --model unet_resnet768 \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 56 \
#    --eval_batch_size 56 \
#    --wandb_run_name liang_unet_resnet768_ddpm_linear \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm

#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --model unet_resnet1024 \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 62 \
#    --eval_batch_size 62 \
#    --wandb_run_name liang_unet_resnet1024_ddpm_linear \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm

#python main.py \
#    --dataset face \
#    --model transformer_2d \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 50 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_ddpm_linear_test \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb