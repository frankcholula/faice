#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --prediction_type v_prediction \
    --image_size 128 \
    --num_epochs 500 \
    --num_train_timesteps 4000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_ablation_vprediction \
    --calculate_fid \
    --calculate_is \
    --verbose
