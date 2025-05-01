#!/bin/bash

# This keeps the depth of base channel and increase the depth from 2 to 4.

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --layers_per_block 4 \
    --base_channels 128 \
    --attention_head_dim 256 \
    --upsample_type conv \
    --downsample_type conv \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name width128_depth4 \
    --calculate_fid \
    --calculate_is \
    --verbose
