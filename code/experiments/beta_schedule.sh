#!/bin/bash

BETA_SETTING=("linear" "squaredcos_cap_v2")

for BETA in "${BETA_SETTING[@]}"; do
    python main.py \
    --dataset face \
    --num_epochs 500 \
    --beta_schedule "$BETA" \
    --batch_size
    --verbose \
    --wandb_run_name "beta_${BETA}" \
    --calculate_fid \
    --calculate_is \
    --no_confrim
done
