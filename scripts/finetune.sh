#!/bin/bash 
#SBATCH --account=aip-lilimou        
#SBATCH --time=2-12:00
#SBATCH --nodes=1                     
#SBATCH --cpus-per-task=6              
#SBATCH --gpus-per-node=4
#SBATCH --mem=96G                      
#SBATCH --job-name=pythia-pile       
#SBATCH --output=logs/slurm-%j.out          


nvidia-smi 
# source preamble.sh 

export HF_HOME=$SCRATCH 
export WANDB_PROJECT="ckpt-compression"
export WANDB_NAME="pythia-pile"
export OUTPUT_DIR=$SCRATCH/runs/$WANDB_NAME

accelerate launch \
    --config_file=accel_configs/fsdp.yaml \
    finetune.py \
        --model=EleutherAI/pythia-410m \
        --task=pile \
        --output_dir=$OUTPUT_DIR \
        --lr=3e-4 \
        --beta1=0.9 \
        --beta2=0.95 \
        --weight_decay=0.1 \
        --eval_every_steps=1000 \