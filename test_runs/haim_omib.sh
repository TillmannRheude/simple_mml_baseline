#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

split_nr=${SLURM_ARRAY_TASK_ID}

batch_size=16
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

modelname_omib_beta=0.0008677777136376928
modelname_omib_cross_attn_network_dim_feedforward=1024
modelname_omib_cross_attn_network_dropout=0.0
modelname_omib_cross_attn_network_num_heads=8
modelname_omib_cross_attn_network_num_layers=16
modelname_omib_warmup_epochs=1

modelname_optimizer_lr=2.6270658624729195e-05
modelname_optimizer_warmup_steps=100
modelname_optimizer_weight_decay=0.001


python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_haim" \
        encoders="mimic_haim" \
        modelname=omib \
        wandb.group="CV-Haim-OMIB" \
        missing.missing_train=[0.0] \
        missing.missing_valid=[0.0] \
        missing.missing_test=[0.0] \
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
        modelname.omib.beta=${modelname_omib_beta} \
        modelname.omib.cross_attn_network.dim_feedforward=${modelname_omib_cross_attn_network_dim_feedforward} \
        modelname.omib.cross_attn_network.dropout=${modelname_omib_cross_attn_network_dropout} \
        modelname.omib.cross_attn_network.num_heads=${modelname_omib_cross_attn_network_num_heads} \
        modelname.omib.cross_attn_network.num_layers=${modelname_omib_cross_attn_network_num_layers} \
        modelname.omib.warmup_epochs=${modelname_omib_warmup_epochs} \