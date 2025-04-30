#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --model lant_unet \
    --pipeline ldmp \
    --scheduler ddim \
    --beta_schedule scaled_linear \
    --image_size 128 \
    --num_epochs 1 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1000 \
    --train_batch_size 96 \
    --eval_batch_size 96 \
    --wandb_run_name liang_ldmp_ddim_linear \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb