set +e
source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate faice
export WANDB_ENTITY=frankcholula

    for prediction_type in v_prediction epsilon; do
        for rescale_betas_zero_snr in True False; do 
            python main.py \
                --dataset face \
                --pipeline ddim \
                --scheduler ddim \
                --beta_schedule linear \
                --prediction_type $prediction_type \
                --rescale_betas_zero_snr $rescale_betas_zero_snr \
                --image_size 128 \
                --num_epochs 50 \
                --train_batch_size 64 \
                --eval_batch_size 64 \
                --calculate_fid \
                --calculate_is \
                --no_confirm \
                --wandb_run_name Ziyu_ddim_50_${prediction_type}_0SNR${rescale_betas_zero_snr} \
        done
    done


echo -e "\n[INFO] Script finished. Dropping into interactive shell..."
exec bash

# python main.py \
#     --dataset face \
#     --pipeline ddim \
#     --scheduler ddim \
#     --beta_schedule squaredcos_cap_v2 \
#     --image_size 128 \
#     --num_epochs 500 \
#     --train_batch_size 64 \
#     --eval_batch_size 64 \
#     --calculate_fid \
#     --calculate_is \
#     --wandb_run_name Ziyu_ddim_cos_v2 \
#     # --no_wandb