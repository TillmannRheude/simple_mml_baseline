#!/bin/bash

#SBATCH --job-name=SB-CV-CREMAD-MBT
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

# modelname.optimizer.lr=0.018733077313616005 modelname.optimizer.warmup_steps=1000 modelname.optimizer.weight_decay=0.1 modelname.bottleneck.dim_feedforward=256 modelname.bottleneck.dropout=0 modelname.bottleneck.layers=2 modelname.bottleneck.nhead=1 modelname.bottleneck.num_bottlenecks=2

modelname_optimizer_lr=0.018733077313616005
modelname_optimizer_warmup_steps=1000
modelname_optimizer_weight_decay=0.1
modelname_bottleneck_dim_feedforward=256
modelname_bottleneck_dropout=0
modelname_bottleneck_layers=2
modelname_bottleneck_nhead=1
modelname_bottleneck_num_bottlenecks=2

batch_size=128

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="crema_d" \
        encoders="crema_d" \
        modelname=mbt \
        wandb.group="CV-CREMAD-MBT" \
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
        modelname.bottleneck.dim_feedforward=${modelname_bottleneck_dim_feedforward} \
        modelname.bottleneck.dropout=${modelname_bottleneck_dropout} \
        modelname.bottleneck.layers=${modelname_bottleneck_layers} \
        modelname.bottleneck.nhead=${modelname_bottleneck_nhead} \
        modelname.bottleneck.num_bottlenecks=${modelname_bottleneck_num_bottlenecks}