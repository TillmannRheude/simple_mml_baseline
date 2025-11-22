#!/bin/bash

#SBATCH --job-name=SB-CV-CREMAD-RegBN
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

modelname_head_transformer_d_model=64
modelname_head_transformer_dim_feedforward=1024
modelname_head_transformer_dropout=0.1
modelname_head_transformer_nhead=4
modelname_head_transformer_num_layers=4

# modelname.optimizer.lr=0.09700202970796584 modelname.optimizer.warmup_steps=1000 modelname.optimizer.weight_decay=0.1 modelname.pipeline.full_seq=True modelname.rbn.affine=True modelname.rbn.momentum=0.05 modelname.rbn.reference_modality_idx=1 modelname.rbn.sigma_MIN=0.01 modelname.rbn.sigma_THR=0.1

modelname_optimizer_lr=0.09700202970796584
modelname_optimizer_warmup_steps=1000
modelname_optimizer_weight_decay=0.1
modelname_rbn_affine=True
modelname_rbn_momentum=0.05
modelname_rbn_reference_modality_idx=1
modelname_rbn_sigma_MIN=0.01
modelname_rbn_sigma_THR=0.1

batch_size=128

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="crema_d" \
        encoders="crema_d" \
        modelname=regbn \
        wandb.group="CV-CREMAD-RegBN" \
        missing.missing_train=[0.0] \
        missing.missing_valid=[0.0] \
        missing.missing_test=[0.0] \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        batch_size=${batch_size} \
        modelname.head_transformer.d_model=${modelname_head_transformer_d_model} \
        modelname.head_transformer.dim_feedforward=${modelname_head_transformer_dim_feedforward} \
        modelname.head_transformer.dropout=${modelname_head_transformer_dropout} \
        modelname.head_transformer.nhead=${modelname_head_transformer_nhead} \
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers} \
        modelname.rbn.affine=${modelname_rbn_affine} \
        modelname.rbn.momentum=${modelname_rbn_momentum} \
        modelname.rbn.reference_modality_idx=${modelname_rbn_reference_modality_idx} \
        modelname.rbn.sigma_MIN=${modelname_rbn_sigma_MIN} \
        modelname.rbn.sigma_THR=${modelname_rbn_sigma_THR}