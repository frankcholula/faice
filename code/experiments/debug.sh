#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --image_size 128 \
    --num_epochs 1 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name ai_surrey_debug
