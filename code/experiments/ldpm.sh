#!/bin/bash

wandb sync --include-synced --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model unet \
    --pipeline ldpm \
    --scheduler ddim \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_ldpm_ddim_linear \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb