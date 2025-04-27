#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp


python main.py \
    --dataset face \
    --model vqvae \
    --pipeline vqvae \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 60 \
    --eval_batch_size 60 \
    --wandb_run_name liang_vqvae \
    --no_confirm \
    --no_wandb