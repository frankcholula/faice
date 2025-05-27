#!/bin/bash

python main.py \
    --dataset face \
    --scheduler pndm \
    --pipeline pndm \
    --num_inference_steps 50 \
    --beta_schedule scaled_linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --prediction_type v_prediction \
    --rescale_betas_zero_snr \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --wandb_run_name task4_pndm_50_v3_scaled_linear_v3 \
    --calculate_fid \
    --calculate_is \
    --verbose
