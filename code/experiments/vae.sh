#!/bin/bash

python main.py \
    --dataset face \
    --model vae \
    --pipeline vae \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 30 \
    --eval_batch_size 30 \
    --wandb_run_name liang_vae \
    --no_confirm \
    --no_wandb