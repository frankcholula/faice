#!/bin/bash

#python main.py \
#    --dataset face \
#    --pipeline consistency \
#    --scheduler ddpm \
#    --beta_schedule linear \
#    --image_size 128 \
#    --num_epochs 500 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --wandb_run_name liang_consistency_ddpm_linear \
#    --calculate_fid \
#    --calculate_is \
#    --no_confirm

python main.py \
    --dataset face \
    --pipeline consistency \
    --scheduler CMStochastic \
    --model unet_resnet \
    --image_size 128 \
    --num_epochs 50 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_timesteps 1000 \
    --num_inference_steps 1 \
    --wandb_run_name liang_unet_resnet512_consistency_CMStochastic_train1000 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb

