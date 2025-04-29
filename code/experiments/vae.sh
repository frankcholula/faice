#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model vae \
    --pipeline vae \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 28 \
    --eval_batch_size 28 \
    --wandb_run_name liang_vae_batch_size_28 \
    --calculate_fid \
    --no_confirm
#    --no_wandb