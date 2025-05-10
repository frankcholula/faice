#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --eta 0.5 \
    --num_inference_steps 100 \
    --beta_schedule squaredcos_cap_v2 \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name task4_ddim_cos_0.5eta \
    --calculate_fid \
    --calculate_is \
    --verbose