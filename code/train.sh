python main.py \
    --dataset face \
    --scheduler ddpm \
    --image_size 128 \
    --num_epochs 1 \
    --train_batch_size 32 \
    --gblur \
    #--beta_schedule linear \


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