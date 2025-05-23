#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule squaredcos_cap_v2 \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_beta_cosine \
    --calculate_fid \
    --calculate_is \
    --verbose
