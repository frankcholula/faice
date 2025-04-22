#!/bin/bash

#SBATCH --partition=debug                   #Selecting partition based on GPU type
#SBATCH --job-name="diffusion_debug"        #Name of Jobs (displayed in squeue)
#SBATCH --nodes=1                           #No of nodes to run job
#SBATCH --ntasks-per-node=10                #No of cores to use per node
#SBATCH --time=00:00:30                     #Maximum time for job to run
#SBATCH --mem=2G                            #Amount of memory per node
#SBATCH --gpus=2                            #Selecting 2 x GPUs
#SBATCH --output=slurm.%N.%j.out            #Output file for stdout (optional)
#SBATCH --error=slurm.%N.%j.err             #Error file for stderr (optional)

python main.py \
    --dataset face \
    --scheduler ddpm \
    --beta_schedule linear \
    --model unet \
    --image_size 128 \
    --num_epochs 1 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --wandb_run_name ai_surrey_debug
