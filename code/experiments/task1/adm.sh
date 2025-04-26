#!/bin/bash

# This is after all the optimizations for ADM. 
python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name task1_adm \
    --calculate_fid \
    --calculate_is \
    --verbose
