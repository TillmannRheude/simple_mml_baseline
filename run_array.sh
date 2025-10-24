#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=shard:2 
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

# The array index will be used as the split_nr
split_nr=${SLURM_ARRAY_TASK_ID}

python3 main.py modelname=transformer split_nr=${split_nr} wandb.group="CV-Transformer"