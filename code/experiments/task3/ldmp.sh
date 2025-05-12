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
#    --model l_unet_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_l_unet_block_5_head_dim_64_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.1_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model l_unet_block_5_head_dim_64_layer_4 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_l_unet_block_5_head_dim_64_layer_4_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.1_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm \
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model unet_l_block_6_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_6_head_dim_64_xformers_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model unet_xl_block_6_head_dim_64_layer_4 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_xl_block_6_head_dim_64_layer_4_xformers_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --eta 0.5 \
#    --RHFlip \
#    --center_crop_arr \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_train_eta_0.5_RHFlip_center_crop_arr_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --eta 0.5 \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_train_eta_0.5_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --attention_head_dim 64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --eta 0.5 \
#    --RHFlip \
#    --center_crop_arr \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 4000 \
#    --num_inference_steps 4000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_RHFlip_center_crop_arr_eta0.5_4000_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4_ag_bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


python main.py \
    --dataset face \
    --model unet_l_block_5_head_dim_64 \
    --attention_head_dim 64 \
    --pipeline ldmp \
    --scheduler ddim \
    --beta_schedule scaled_linear \
    --eta 0.5 \
    --RHFlip \
    --center_crop_arr \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --num_inference_steps 100 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --loss_type mse \
    --use_lpips \
    --lpips_net alex \
    --lpips_weight 0.05 \
    --wandb_run_name liang_unet_l_block_5_head_dim_64_RHFlip_eta0.5_4000_100_ldmp_ddim_scaled_linear_lpips_0.05vqvae3_loss_weight0.4_ag_bs16_bs64 \
    --calculate_fid \
    --calculate_is \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --attention_head_dim 64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --eta 0.5 \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 100 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --use_lpips \
#    --lpips_net alex \
#    --lpips_weight 0.05 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_ldmp_ddim_scaled_linear_eta0.5_timesteps1000_100_lpips_0.05_vqvae3_loss_weight0.4_ag_bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --attention_head_dim 64 \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 4000 \
#    --num_inference_steps 100 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_train_timesteps_4000_100_ldmp_ddim_scaled_linear_vqvae3_loss_weight0.4_ag_bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb

#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler pndm \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 50 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_unet_l_block_5_head_dim_64_pndm_scaled_linear_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model unet \
#    --unet_variant adm \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 4000 \
#    --num_inference_steps 4000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_adm_ddim_scaled_linear_train_steps4000_vqvae3_loss_weight0.4bs16_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --no_confirm
#    --no_wandb


#python main.py \
#    --dataset face \
#    --model unet_l_block_5_head_dim_64 \
#    --pipeline ldmp \
#    --scheduler ddim \
#    --eta 0.5 \
#    --attention_head_dim 64 \
#    --beta_schedule scaled_linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 1000 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_ldmp_ddim_scaled_linear_pretrain_vqvae_bs64 \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm
#    --no_wandb