#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --pipeline ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 1 \
    --num_train_timesteps 1000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --loss_type mse \
    --use_lpips \
    --lpips_net vgg \
    --lpips_weight 0.1 \
    --wandb_run_name task6_mse_lpips_vgg_0.1 \
    --calculate_fid \
    --calculate_is \
    --verbose    
