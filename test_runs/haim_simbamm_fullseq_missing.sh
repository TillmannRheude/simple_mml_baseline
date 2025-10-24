#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

split_nr=${SLURM_ARRAY_TASK_ID}

batch_size=128
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
modelname_optimizer_lr=3.034921659273387e-05
modelname_optimizer_warmup_steps=0
modelname_optimizer_weight_decay=0.0

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_haim" \
        encoders="mimic_haim" \
        modelname=transformer \
        wandb.group="CV-Haim-SimBaMM-Full-Missing" \
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
        modelname.pipeline.full_seq=True \
        modelname.head_transformer.d_model=${modelname_head_transformer_d_model} \
        modelname.head_transformer.dim_feedforward=${modelname_head_transformer_dim_feedforward} \
        modelname.head_transformer.dropout=${modelname_head_transformer_dropout} \
        modelname.head_transformer.nhead=${modelname_head_transformer_nhead} \
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers} \