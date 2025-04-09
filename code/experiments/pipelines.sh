#!/bin/bash

#python main.py \
#    --dataset face \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_ddpm_linear \
#    --calculate_fid \
#    --calculate_is

python main.py \
    --dataset face \
    --pipeline consistency \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_consistency_ddpm_linear \
    --calculate_fid \
    --calculate_is \
    --RHFlip


