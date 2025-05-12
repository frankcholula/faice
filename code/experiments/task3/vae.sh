#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

#python main.py \
#    --dataset face \
#    --model vae_b_16 \
#    --pipeline vae \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 112 \
#    --eval_batch_size 112 \
#    --wandb_run_name liang_vae_latent_channels16_batch_size_112 \
#    --calculate_fid \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model vae_l_4 \
#    --pipeline vae \
#    --image_size 128 \
#    --num_epochs 1 \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --wandb_run_name liang_vae_l_4_latent_channels4_loss_weight_0.5_batch_size_16 \
#    --calculate_fid \
#    --no_confirm \
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model vae_b_4 \
#    --pipeline vae \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --wandb_run_name liang_vae_b_4_batch_loss_weight_0.05_size_16 \
#    --calculate_fid \
#    --no_confirm
#    --no_wandb

python main.py \
    --dataset face \
    --model vae_l_4 \
    --pipeline vae \
    --RHFlip \
    --center_crop_arr \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name liang_vae_l_4_batch_loss_weight_RHFlip_center_crop_0.05_size_16 \
    --calculate_fid \
    --no_confirm
#    --no_wandb