#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

#python main.py \
#    --dataset face_dialog \
#    --model unet_cond_l_block_4 \
#    --pipeline stable_diffusion \
#    --attention_head_dim 64 \
#    --scheduler pndm \
#    --beta_schedule scaled_linear \
#    --RHFlip \
#    --center_crop_arr \
#    --image_size 256 \
#    --num_epochs 200 \
#    --num_train_timesteps 1000 \
#    --num_inference_steps 999 \
#    --train_batch_size 32 \
#    --eval_batch_size 32 \
#    --wandb_run_name liang_totally_fine_tuning_stable_diffusion_pndm_scaled_linear_bs32 \
#    --calculate_fid \
#    --calculate_is \
#    --enable_xformers_memory_efficient_attention \
#    --allow_tf32 \
#    --use_ema \
#    --no_confirm
#    --no_wandb


python main.py \
    --dataset face_dialog \
    --model unet_cond_l_block_4 \
    --pipeline stable_diffusion \
    --attention_head_dim 64 \
    --scheduler pndm \
    --beta_schedule scaled_linear \
    --RHFlip \
    --center_crop_arr \
    --image_size 256 \
    --num_epochs 200 \
    --num_train_timesteps 1000 \
    --num_inference_steps 999 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --wandb_run_name liang_freeze3layers_stable_diffusion_pndm_scaled_linear_bs32 \
    --calculate_fid \
    --calculate_is \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    --use_ema \
    --no_confirm \
    --no_wandb