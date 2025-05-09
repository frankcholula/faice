#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp


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

#python main.py \
#    --dataset face \
#    --model DiT_B_2_vae_channels_4\
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 152 \
#    --eval_batch_size 152 \
#    --wandb_run_name liang_DiT_B_2_vae_channels_4_ddim_scaled_linear_bs152 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model DiT_B_2_vae_channels_4\
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_DiT_B_2_vae_channels_4_ddim_scaled_linear_bs64_vae0.1 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model DiT_B_2_vae_channels_16\
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_DiT_B_2_vae_channels_16_ddim_scaled_linear_bs64_vae0.002 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model DiT_B_2_vae_channels_4\
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_DiT_B_2_vae_channels_4_ddim_scaled_linear_bs64_vae_l_4_0.05 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model DiT_B_2_vae_channels_4_compress_4\
#    --pipeline dit_vae \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 16 \
#    --eval_batch_size 16 \
#    --wandb_run_name liang_DiT_B_2_vae_channels_4_compress_4_ddim_scaled_linear_bs16_vae0.05 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


python main.py \
    --dataset face \
    --model DiT_B_2_vae_channels_4\
    --attention_head_dim 64 \
    --pipeline dit_vae \
    --scheduler ddim \
    --beta_schedule scaled_linear \
    --image_size 256 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 8 \
    --eval_batch_size 8 \
    --wandb_run_name liang_pre_train_vae_channels_4_ddim_scaled_linear_bs64_vae_l_4_0.05 \
    --calculate_fid \
    --calculate_is \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    --no_confirm \
    --no_wandb


