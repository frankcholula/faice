#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

#python main.py \
#    --dataset face \
#    --model transformer_2d \
#    --pipeline dit \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 160 \
#    --eval_batch_size 160 \
#    --wandb_run_name liang_transformer_2d_ddpm_linear \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb


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


python main.py \
    --dataset face \
    --model DiT_B_4 \
    --pipeline dit \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_DiT_B_4_ddpm_linear_bs16 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb


