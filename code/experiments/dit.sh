#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model transformer_2d_xformers_fast \
    --pipeline dit \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 1 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 164 \
    --eval_batch_size 164 \
    --wandb_run_name liang_transformer_2d_xformers_fast_ddpm_linear \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb

#python main.py \
#    --dataset face \
#    --model transformer_2d \
#    --pipeline dit \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 1 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 128 \
#    --eval_batch_size 128 \
#    --wandb_run_name liang_dit_ddpm_linear_batch_size_128 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

