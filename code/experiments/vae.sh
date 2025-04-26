#!/bin/bash

python main.py \
    --dataset face \
    --model vae \
    --pipeline vae \
    --image_size 128 \
    --num_epochs 20 \
    --train_batch_size 60 \
    --eval_batch_size 60 \
    --wandb_run_name liang_vae \
    --no_confirm \
    --no_wandb