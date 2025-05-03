#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

#python main.py \
#    --dataset face \
#    --model dit_transformer \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 20 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 40 \
#    --eval_batch_size 40 \
#    --wandb_run_name liang_dit_vae_ddim_scaled_linear_batch_size_40_test \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model transformer_2d_xformers_vae \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 300 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_dit_vae_ddim_scaled_linear_batch_size_64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model transformer_2d_xformers_vae \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --wandb_run_name liang_transformer_2d_xformers_vae_ddim_linear_pretrain_vae \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model transformer_2d_xformers_vae \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 200 \
#    --eval_batch_size 200 \
#    --wandb_run_name liang_transformer_2d_vae_ddim_linear_vae16 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model dit_transformer \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 164 \
#    --eval_batch_size 164 \
#    --wandb_run_name liang_dit_transformer_ddim_linear_vae16 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model dit_transformer \
#    --pipeline dit_vae \
#    --scheduler dpmsolvermultistep \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 1 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 164 \
#    --eval_batch_size 164 \
#    --wandb_run_name liang_dit_transformer_dpmsolvermultistep_linear_vae16 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb

python main.py \
    --dataset face \
    --model dit_transformer \
    --pipeline dit_vae \
    --scheduler ddim \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 128 \
    --eval_batch_size 128 \
    --wandb_run_name liang_dit_transformer_ddim_linear_vae16_bs_82 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb

#python main.py \
#    --dataset face \
#    --model transformer_2d_xformers_vae \
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 8 \
#    --eval_batch_size 8 \
#    --wandb_run_name liang_transformer_2d_xformers_vae_ddim_linear_vae4_bs82 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb


