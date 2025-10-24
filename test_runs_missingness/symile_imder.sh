#!/bin/bash

#SBATCH --job-name=SB-MR
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5 # 5 splits * 4 missing rates

# The array index will be used to determine split_nr and missing rate
task_id=${SLURM_ARRAY_TASK_ID}

# 5 splits for cross-validation
split_nr=$(( (task_id - 1) % 5 + 1 ))

# 4 different missing rates
missing_rates=(0.5)
rate_index=$(( (task_id - 1) / 5 ))
rate=${missing_rates[$rate_index]}

batch_size=128  # 32, 128 just for training time logging
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

modelname_optimizer_lr=7.635783605901435e-05
modelname_optimizer_warmup_steps=1000
modelname_optimizer_weight_decay=0.01

modelname_ddpms_dropout=0
modelname_ddpms_hidden_dim=256
modelname_ddpms_n_steps=100
modelname_ddpms_nhead=8
modelname_ddpms_num_layers=3
modelname_imder_beta=0.03973810545828186


python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mimic_symile" \
        encoders="mimic_symile" \
        modelname=imder \
        wandb.group="CV-Symile-IMDer-Missing-${rate}" \
        missing.missing_train=[${rate},${rate}] \
        missing.missing_valid=[${rate},${rate}] \
        missing.missing_test=[${rate},${rate}] \
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
        modelname.imder.beta=${modelname_imder_beta} \
        modelname.ddpms.n_steps=${modelname_ddpms_n_steps} \
        modelname.ddpms.hidden_dim=${modelname_ddpms_hidden_dim} \
        modelname.ddpms.num_layers=${modelname_ddpms_num_layers} \
        modelname.ddpms.dropout=${modelname_ddpms_dropout} \
        modelname.ddpms.nhead=${modelname_ddpms_nhead}
