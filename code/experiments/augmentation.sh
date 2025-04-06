python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --gblur \
    --wandb_run_name ddpm-linear-face-129-gblur
