#!/bin/bash
#SBATCH --job-name=SimBas
#SBATCH -p agpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00

python3 main.py \
    modelname="transformer"