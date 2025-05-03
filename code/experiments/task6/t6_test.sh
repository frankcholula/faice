#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddim \
    --pipeline ddim \
    --eta 0.5 \
    --num_inference_steps 100 \
    --beta_schedule linear \
    --model unet \
    --unet_variant adm \
    --attention_head_dim 64 \
    --image_size 128 \
    --num_epochs 1 \
    --num_train_timesteps 1000 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --loss_type l1 \
    --use_lpips True \
    --lpips_net vgg \
    --lpips_weight 0.2 \
    --wandb_run_name task6_test \
    --calculate_fid \
    --calculate_is \
    --verbose
