#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

# The array index will be used as the split_nr
split_nr=${SLURM_ARRAY_TASK_ID}

lr=0.00019852737148274023
warmup_steps=200
weight_decay=0.0
loss_alpha=0.01336717903726672
loss_beta=0.09249944162004872

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_symile" \
        encoders="mimic_symile" \
        modelname=shaspec \
        wandb.group="CV-Symile-Shaspec" \
        batch_size=128 \
        missing.missing_train='[0.15,0.15]' \
        missing.missing_valid='[0.15,0.15]' \
        missing.missing_test='[0.15,0.15]' \
        modelname.optimizer.lr=${lr} \
        modelname.optimizer.warmup_steps=${warmup_steps} \
        modelname.optimizer.weight_decay=${weight_decay} \
        modelname.shaspec.loss_alpha=${loss_alpha} \
        modelname.shaspec.loss_beta=${loss_beta}
