#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --wandb_run_name task1_adm \
    --calculate_fid \
    --calculate_is \
    --verbose
