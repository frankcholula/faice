#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

#python main.py \
#    --dataset face \
#    --model latent_unet \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 96 \
#    --eval_batch_size 96 \
#    --wandb_run_name liang_ldmp_ddim_scaled_linear_pretrain_vqvae_bs96 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_for_ldm \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 96 \
#    --eval_batch_size 96 \
#    --wandb_run_name liang_ldmp_fefault_unet_for_ldm_ddim_scaled_linear_pretrain_vqvae \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model latent_unet \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 100 \
#    --eval_batch_size 100 \
#    --wandb_run_name liang_ldmp_ddim_scaled_linear_vqvae32_bs100 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model latent_unet_xl \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_latent_unet_xl_no_resnet_ldmp_ddim_scaled_linear_pretrain_vqvae_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model latent_unet \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_latent_unet_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.5_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model latent_unet_xl \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_latent_unet_xl_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.1_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

python main.py \
    --dataset face \
    --model unet_pp \
    --pipeline ldmp \
    --scheduler ddim \
    --beta_schedule scaled_linear \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_unet_pp_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.1_bs64 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb