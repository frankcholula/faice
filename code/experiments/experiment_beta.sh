#!/bin/bash

# BETA_SETTING override
BETA_SETTING=("linear" "squaredcos_cap_v2")

# Loop through the beta schedule settings
for BETA in "${BETA_SETTING[@]}"; do
    python main.py \
    --dataset face \
    --num_epochs 500 \
    --beta_scheduler "$BETA" \
    --verbose \
    --wandb_run_name "emre_face_beta_${BETA}_test" \
    --output_dir "output/face_beta_${BETA}_test" \
done
