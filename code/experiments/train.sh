python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 1 \
    --train_batch_size 32 \
    --gblur \
    --wandb_run_name ziyu-test-sh-augmentation
