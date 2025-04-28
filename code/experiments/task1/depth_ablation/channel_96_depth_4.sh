#!/bin/bash

# This decraeses the base channel from 128 to 96 and increase the depth from 2 to 4.
# The rest of the configuration is the same as the ddpm_base.sh file.

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --layers_per_block 4 \
    --base_channels 96 \
    --attention_head_dim 192 \
    --upsample_type conv \
    --downsample_type conv \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name task1_ddpm_width96_depth4_ablation \
    --calculate_fid \
    --calculate_is \
    --verbose
