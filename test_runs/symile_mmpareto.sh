#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

# The array index will be used as the split_nr
split_nr=${SLURM_ARRAY_TASK_ID}

batch_size=128
encoders_sequential1_transformer_dropout=0.2
encoders_sequential1_transformer_intermediate_size=2048
encoders_sequential1_transformer_num_attention_heads=16
encoders_sequential1_transformer_num_hidden_layers=6
encoders_sequential2_mlp_hidden_dims="[1024,2048,1024]"
encoders_sequential2_mlp_hidden_dropouts="[0.2,0.2,0.2]"
encoders_vision_vit_dropout=0.2
encoders_vision_vit_vit=vit_b_32
modelname_head_transformer_d_model=128
modelname_head_transformer_dim_feedforward=512
modelname_head_transformer_dropout=0.2
modelname_head_transformer_nhead=4
modelname_head_transformer_num_layers=8

modelname_optimizer_lr=0.0001922967137500355
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0.01

modelname_mmpareto_gamma=0.10121853219084756
modelname_mmpareto_unimodal_loss_weight=0.07412787043906463


python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_symile" \
        encoders="mimic_symile" \
        modelname=mmpareto \
        wandb.group="CV-Symile-MMPareto" \
        missing.missing_train='[0.0,0.0]' \
        missing.missing_valid='[0.0,0.0]' \
        missing.missing_test='[0.0,0.0]' \
        batch_size=${batch_size} \
        encoders.sequential1.transformer.dropout=${encoders_sequential1_transformer_dropout} \
        encoders.sequential1.transformer.intermediate_size=${encoders_sequential1_transformer_intermediate_size} \
        encoders.sequential1.transformer.num_attention_heads=${encoders_sequential1_transformer_num_attention_heads} \
        encoders.sequential1.transformer.num_hidden_layers=${encoders_sequential1_transformer_num_hidden_layers} \
        encoders.sequential2.mlp.hidden_dims=${encoders_sequential2_mlp_hidden_dims} \
        encoders.sequential2.mlp.hidden_dropouts=${encoders_sequential2_mlp_hidden_dropouts} \
        encoders.vision.vit.dropout=${encoders_vision_vit_dropout} \
        encoders.vision.vit.vit=${encoders_vision_vit_vit} \
        modelname.head_transformer.d_model=${modelname_head_transformer_d_model} \
        modelname.head_transformer.dim_feedforward=${modelname_head_transformer_dim_feedforward} \
        modelname.head_transformer.dropout=${modelname_head_transformer_dropout} \
        modelname.head_transformer.nhead=${modelname_head_transformer_nhead} \
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers} \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        modelname.mmpareto.gamma=${modelname_mmpareto_gamma} \
        modelname.mmpareto.unimodal_loss_weight=${modelname_mmpareto_unimodal_loss_weight}
