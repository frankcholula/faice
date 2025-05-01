#!/bin/bash


python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --multi_res \
    --attention_head_dim 256 \
    --upsample_type resnet \
    --downsample_type resnet \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 24 \
    --eval_batch_size 24 \
    --wandb_run_name multi_res_resnet_sampling \
    --calculate_fid \
    --calculate_is \
    --verbose
