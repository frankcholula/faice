#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --pipeline ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --loss_type hybrid \
    --hybrid_weight 0.75 \
    --wandb_run_name task6_hybrid_0.75 \
    --calculate_fid \
    --calculate_is \
    --verbose