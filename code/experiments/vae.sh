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

python main.py \
    --dataset face \
    --model vae_l_4 \
    --pipeline vae \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 112 \
    --eval_batch_size 112 \
    --wandb_run_name liang_vae_l_4_latent_channels4_loss_weight_0.1_batch_size_112 \
    --calculate_fid \
    --no_confirm
#    --no_wandb