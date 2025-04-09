#!/bin/bash

BETA_SETTING=("linear" "squaredcos_cap_v2")

for BETA in "${BETA_SETTING[@]}"; do
    python main.py \
    --dataset face \
    --scheduler "ddpm" \
    --beta_schedule "$BETA" \
    --num_epochs 500 \
    --learning_rate 2e-4 \
    --beta_schedule "$BETA" \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --image_size 128 \
    --wandb_run_name "ddpm_${BETA}" \
    --calculate_fid \
    --calculate_is \
    --verbose \
    --no_confirm
done
