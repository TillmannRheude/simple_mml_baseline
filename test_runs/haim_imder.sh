#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

split_nr=${SLURM_ARRAY_TASK_ID}

batch_size=32
encoders_sequential_transformer_dropout=0.2
encoders_sequential_transformer_intermediate_size=1024
encoders_sequential_transformer_num_attention_heads=8
encoders_sequential_transformer_num_hidden_layers=2
encoders_vision_vit_dropout=0.0
encoders_vision_vit_vit=vit_b_32
modelname_head_transformer_d_model=256
modelname_head_transformer_dim_feedforward=256
modelname_head_transformer_dropout=0.1
modelname_head_transformer_nhead=4
modelname_head_transformer_num_layers=4

modelname_ddpms_d_model=256
modelname_ddpms_dropout=0.1
modelname_ddpms_hidden_dim=128
modelname_ddpms_n_steps=10
modelname_ddpms_nhead=2
modelname_ddpms_num_layers=7

modelname_optimizer_lr=1.9321383181021892e-05
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0.01


python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_haim" \
        encoders="mimic_haim" \
        modelname=imder \
        wandb.group="CV-Haim-IMDer" \
        missing.missing_train=[0.3] \
        missing.missing_valid=[0.3] \
        missing.missing_test=[0.3] \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        batch_size=${batch_size} \
        encoders.sequential.transformer.dropout=${encoders_sequential_transformer_dropout} \
        encoders.sequential.transformer.intermediate_size=${encoders_sequential_transformer_intermediate_size} \
        encoders.sequential.transformer.num_attention_heads=${encoders_sequential_transformer_num_attention_heads} \
        encoders.sequential.transformer.num_hidden_layers=${encoders_sequential_transformer_num_hidden_layers} \
        encoders.vision.vit.dropout=${encoders_vision_vit_dropout} \
        encoders.vision.vit.vit=${encoders_vision_vit_vit} \
        modelname.head_transformer.d_model=${modelname_head_transformer_d_model} \
        modelname.head_transformer.dim_feedforward=${modelname_head_transformer_dim_feedforward} \
        modelname.head_transformer.dropout=${modelname_head_transformer_dropout} \
        modelname.head_transformer.nhead=${modelname_head_transformer_nhead} \
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers} \
        modelname.ddpms.d_model=${modelname_ddpms_d_model} \
        modelname.ddpms.dropout=${modelname_ddpms_dropout} \
        modelname.ddpms.hidden_dim=${modelname_ddpms_hidden_dim} \
        modelname.ddpms.n_steps=${modelname_ddpms_n_steps} \
        modelname.ddpms.nhead=${modelname_ddpms_nhead} \
        modelname.ddpms.num_layers=${modelname_ddpms_num_layers} \