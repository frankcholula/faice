#!/bin/bash

#python main.py \
#    --dataset face \
#    --scheduler ddpm.sh \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_gblur_ddpm_linear \
#    --calculate_fid \
#    --calculate_is \
#    --gblur


python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet_resnet \
    --image_size 128 \
    --num_epochs 10 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_unet_resnet_ddpm_linear \
    --calculate_fid \
    --calculate_is \
    --no_wandb