#!/bin/bash

#SBATCH --job-name=SB-CV-CREMAD-OMIB
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

# modelname.omib.beta=0.0004761148794245736 modelname.omib.cross_attn_network.dim_feedforward=1024 modelname.omib.cross_attn_network.dropout=0.2 modelname.omib.cross_attn_network.num_heads=2 modelname.omib.cross_attn_network.num_layers=1 modelname.omib.warmup_epochs=1 modelname.optimizer.lr=0.01606524607052335 modelname.optimizer.warmup_steps=100 modelname.optimizer.weight_decay=0.1

modelname_optimizer_lr=0.01606524607052335
modelname_optimizer_warmup_steps=100
modelname_optimizer_weight_decay=0.1
modelname_omib_beta=0.0004761148794245736
modelname_omib_cross_attn_network_dim_feedforward=1024
modelname_omib_cross_attn_network_dropout=0.2
modelname_omib_cross_attn_network_num_heads=2
modelname_omib_cross_attn_network_num_layers=1
modelname_omib_warmup_epochs=1

batch_size=128

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="crema_d" \
        encoders="crema_d" \
        modelname=omib \
        wandb.group="CV-CREMAD-OMIB" \
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
        modelname.omib.beta=${modelname_omib_beta} \
        modelname.omib.cross_attn_network.dim_feedforward=${modelname_omib_cross_attn_network_dim_feedforward} \
        modelname.omib.cross_attn_network.dropout=${modelname_omib_cross_attn_network_dropout} \
        modelname.omib.cross_attn_network.num_heads=${modelname_omib_cross_attn_network_num_heads} \
        modelname.omib.cross_attn_network.num_layers=${modelname_omib_cross_attn_network_num_layers} \
        modelname.omib.warmup_epochs=${modelname_omib_warmup_epochs}