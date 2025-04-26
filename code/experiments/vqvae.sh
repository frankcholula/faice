#!/bin/bash

python main.py \
    --dataset face \
    --model vqvae \
    --pipeline vqvae \
    --image_size 128 \
    --num_epochs 20 \
    --train_batch_size 12 \
    --eval_batch_size 12 \
    --wandb_run_name liang_vqvae \
    --no_confirm \
    --no_wandb