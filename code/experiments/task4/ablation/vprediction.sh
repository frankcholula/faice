#!/bin/bash

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --unet_variant ddpm \
    --image_size 128 \
    --num_epochs 500 \
    --prediction_type v_prediction \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name task4_ablation_vprediction \
    --calculate_fid \
    --calculate_is \
    --verbose
