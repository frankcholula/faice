#!/bin/bash

python main.py \
    --dataset face \
    --pipeline consistency \
    --scheduler CMStochastic \
    --model unet_resnet \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_timesteps 40 \
    --num_inference_steps 1 \
    --wandb_run_name liang_unet_resnet512_consistency_CMStochastic_default \
    --calculate_fid \
    --calculate_is
