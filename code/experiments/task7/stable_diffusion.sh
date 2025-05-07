#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model unet_l_block_6_head_dim_64 \
    --pipeline stable_diffusion \
    --scheduler pndm \
    --beta_schedule scaled_linear \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_stable_diffusion_pndm_scaled_linear_bs64 \
    --calculate_fid \
    --calculate_is \
    --enable_xformers_memory_efficient_attention \
    --allow_tf32 \
    --use_ema \
    --no_confirm \
    --no_wandb