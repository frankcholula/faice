#!/bin/bash

wandb sync --clean-old-hours 1
wandb artifact cache cleanup 0GB --remove-temp

python main.py \
    --dataset face \
    --pipeline consistency \
    --scheduler CMStochastic \
    --model unet_resnet512 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_train_timesteps 1000 \
    --num_inference_steps 10 \
    --wandb_run_name liang_unet_resnet512_consistency_CMStochastic_train1000_inference10 \
    --calculate_fid \
    --calculate_is \
    --no_confirm \
    --no_wandb


