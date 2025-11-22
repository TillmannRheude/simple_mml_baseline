#!/bin/bash

#SBATCH --job-name=SB-CV-CREMAD-SimBaMM-Full
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

# modelname.head_transformer.d_model=64 modelname.head_transformer.dim_feedforward=1024 modelname.head_transformer.dropout=0.1 modelname.head_transformer.nhead=4 modelname.head_transformer.num_layers=4 modelname.optimizer.lr=0.02946976043941851 modelname.optimizer.warmup_steps=1000 modelname.optimizer.weight_decay=0.01

modelname_head_transformer_d_model=64
modelname_head_transformer_dim_feedforward=1024
modelname_head_transformer_dropout=0.1
modelname_head_transformer_nhead=4
modelname_head_transformer_num_layers=4

modelname_optimizer_lr=0.006132523742119594
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0.1

batch_size=128
modelname_pipeline_full_seq=True

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="crema_d" \
        encoders="crema_d" \
        modelname=transformer \
        wandb.group="CV-CREMAD-SimBaMM-Full" \
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
        modelname.pipeline.full_seq=${modelname_pipeline_full_seq} 