python main.py \
    --dataset face \
    --pipeline ddim \
    --scheduler ddim \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --calculate_fid \
    --calculate_is \
    --wandb_run_name Ziyu_ddim_linear \

python main.py \
    --dataset face \
    --pipeline ddim \
    --scheduler ddim \
    --beta_schedule squaredcos_cap_v2 \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --calculate_fid \
    --calculate_is \
    --wandb_run_name Ziyu_ddim_cos \
    # --no_wandb
