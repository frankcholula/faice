#!/bin/bash

# This changes the up/downsamle to using resnet.
# The rest is the same as the ddpm_base.sh file.
# Run this fifth.

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --attention_head_dim 256 \
    --upsample_type resnet \
    --downsample_type resnet \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name resnet_sampling \
    --calculate_fid \
    --calculate_is \
    --verbose
