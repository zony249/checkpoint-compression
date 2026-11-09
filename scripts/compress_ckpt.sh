#!/bin/bash 
#SBATCH --account=aip-lilimou        
#SBATCH --time=0-03:00
#SBATCH --nodes=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G                      
#SBATCH --job-name=llama-finetune       
#SBATCH --output=logs/slurm-%j.out          


nvidia-smi 
# source preamble.sh 

export HF_HOME=$SCRATCH 
# export WANDB_PROJECT="ckpt-compression"
# export WANDB_NAME="llama-test"

accelerate launch \
    --config_file=accel_configs/fsdp.yaml \
    compress_ckpt.py \
        --base_model=EleutherAI/pythia-410M \
        --checkpoints_dir=/home/zonglin1/scratch/runs/pythia-pile \
        --output_compression=$SCRATCH/runs/pythia-pile-compress \
        --output_recon=$SCRATCH/runs/pythia-pile-recon \