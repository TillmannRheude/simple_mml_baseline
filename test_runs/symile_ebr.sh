#!/bin/bash

#SBATCH --job-name=SimBaMM-EBR
#SBATCH -p pgpu
#SBATCH --gres=gpu:2
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
lr=0.002077800500632696
warmup_steps=100
weight_decay=0.001
momentum=0.99
ebr_alpha=0.0018801716039664616
ebr_interleave_epochs=0
ebr_hidden_dim=2048
ebr_d_shared=128

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_symile" \
        encoders="mimic_symile" \
        modelname=ebr \
        wandb.group="CV-Symile-EBR" \
        batch_size=${batch_size} \
        missing.missing_train='[0.0,0.0]' \
        missing.missing_valid='[0.15,0.15]' \
        missing.missing_test='[0.15,0.15]' \
        modelname.optimizer.lr=${lr} \
        modelname.optimizer.warmup_steps=${warmup_steps} \
        modelname.optimizer.weight_decay=${weight_decay} \
        modelname.optimizer.momentum=${momentum} \
        modelname.ebr.alpha_ebr=${ebr_alpha} \
        modelname.ebr.interleave_epochs=${ebr_interleave_epochs} \
        modelname.ebr.hidden_dim=${ebr_hidden_dim} \
        modelname.ebr.d_shared=${ebr_d_shared} \
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
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers}
