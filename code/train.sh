python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --image_size 128 \
    --num_epochs 500 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name liang_ddpm_linear \


# for gblur in "" "--gblur"; do
#     for RHFlip in "" "--RHFlip"; do

#             # exprriment name generation
#             EXP_NAME="faice"
#             [ ! -z "$scheduler" ] && EXP_NAME+="ddpm"
#             [ ! -z "$gblur" ] && EXP_NAME+="-gblur"
#             [ ! -z "$RHFlip" ] && EXP_NAME+="-RHFlip"


#             echo "Running experiment: $EXP_NAME"

#             python main.py \
#                 --scheduler ddpm \
#                 --beta_schedule linear \
#                 --image_size 128 \
#                 --num_epochs 50 \
#                 --batch_size 64 \
#                 --output_dir $EXP_NAME \
#                 $gblur $RHFlip
#         done
#     done
# done