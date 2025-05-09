#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --beta_schedule squaredcos_cap_v2 \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --RHFlip \
    --center_crop_arr \
    --eta 0.5 \
    --prediction_type v_prediction \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --loss_type mse \
    --use_lpips \
    --lpips_net alex \
    --lpips_weight 0.05 \
    --wandb_run_name fruns_adm_bs16_ddim_0.5_v_mse_lpips_alex_0.05 \
    --calculate_fid \
    --calculate_is \
    --verbose