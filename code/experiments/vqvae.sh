#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp


python main.py \
    --dataset face \
    --model vqvae \
    --pipeline vqvae \
    --image_size 128 \
    --num_epochs 400 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_vqvae_batch_size_64_latent_channels_32 \
    --calculate_fid \
    --no_confirm
#    --no_wandb