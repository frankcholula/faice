#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp


python main.py \
    --dataset face \
    --model vqvae \
    --pipeline vqvae \
    --image_size 128 \
    --num_epochs 1 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name liang_vqvae \
    --calculate_fid \
    --no_confirm \
    --no_wandb