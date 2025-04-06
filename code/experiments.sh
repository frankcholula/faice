#!/bin/bash

# Define parameter combinations, you can add more if needed
# BETA_SCHEDULES can be "linear", "squaredcos_cap_v2", etc.
BETA_SCHEDULES=("linear" "squaredcos_cap_v2")
# BLUR_SETTING can be "True" or "False"
BLUR_SETTING=("True" "False")
# FLIP_SETTING can be "True" or "False"
FLIP_SETTING=("True" "False")

# Loop through all combinations of parameters
for BETA in "${BETA_SCHEDULES[@]}"; do
    for BLUR in "${BLUR_SETTING[@]}"; do
        for FLIP in "${FLIP_SETTING[@]}"; do
            echo "Running with beta schedule = $BETA, blur setting =$BLUR, flip setting = $FLIP"
            python main.py --dataset face --num_epochs 1 --beta_scheduler "$BETA" --gblur "$BLUR" --RHFlip "$FLIP" --verbose
        done
    done
done