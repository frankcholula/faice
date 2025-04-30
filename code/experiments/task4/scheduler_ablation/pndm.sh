#!/bin/bash

python main.py \
    --dataset face \
    --scheduler pndm \
    --pipeline pndm \
    --num_inference_steps 50 \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --prediction_type epsilon \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_pndm_50 \
    --calculate_fid \
    --calculate_is \
    --verbose
