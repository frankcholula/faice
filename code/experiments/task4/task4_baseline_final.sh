#!/bin/bash
# This is after all the optimizations for ADM.
# This should also be the baseline for task 4.
python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name task4_bs16_baseline \
    --calculate_fid \
    --calculate_is \
    --verbose
