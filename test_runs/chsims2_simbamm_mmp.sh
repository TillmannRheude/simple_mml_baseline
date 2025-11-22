#!/bin/bash

#SBATCH --job-name=SB-CV-MMP
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

encoders_audio_transformer_num_hidden_layers=8
encoders_audio_transformer_num_attention_heads=2
encoders_audio_transformer_intermediate_size=1024
encoders_vision_transformer_num_hidden_layers=4
encoders_vision_transformer_num_attention_heads=4
encoders_vision_transformer_intermediate_size=512
modelname_head_transformer_d_model=256
modelname_head_transformer_dim_feedforward=1024
modelname_head_transformer_dropout=0.1
modelname_head_transformer_nhead=16
modelname_head_transformer_num_layers=2

# modelname.mmp.attn_steps.dropout=0 modelname.mmp.attn_steps.nhead=2 modelname.mmp.loss_alignment_alpha=0.07598379819791869 modelname.mmp.num_aggregated_tokens=4 modelname.mmp.proj_mlp.dropout=0.1 modelname.mmp.proj_mlp.hidden_dim=128 modelname.optimizer.lr=4.041159777947827e-05 modelname.optimizer.warmup_steps=200 modelname.optimizer.weight_decay=0

modelname_optimizer_lr=4.041159777947827e-05
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0
modelname_mmp_attn_steps_dropout=0
modelname_mmp_attn_steps_nhead=2
modelname_mmp_loss_alignment_alpha=0.07598379819791869
modelname_mmp_num_aggregated_tokens=4
modelname_mmp_proj_mlp_dropout=0.1
modelname_mmp_proj_mlp_hidden_dim=128

batch_size=16

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="ch_sims_v2" \
        encoders="ch_sims_v2" \
        modelname=mmp \
        wandb.group="CV-CHSims2-SimBaMM-MMP" \
        missing.missing_train=[0.15,0.15] \
        missing.missing_valid=[0.15,0.15] \
        missing.missing_test=[0.15,0.15] \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        batch_size=${batch_size} \
        encoders.audio.transformer.intermediate_size=${encoders_audio_transformer_intermediate_size} \
        encoders.audio.transformer.num_attention_heads=${encoders_audio_transformer_num_attention_heads} \
        encoders.audio.transformer.num_hidden_layers=${encoders_audio_transformer_num_hidden_layers} \
        encoders.vision.transformer.intermediate_size=${encoders_vision_transformer_intermediate_size} \
        encoders.vision.transformer.num_attention_heads=${encoders_vision_transformer_num_attention_heads} \
        encoders.vision.transformer.num_hidden_layers=${encoders_vision_transformer_num_hidden_layers} \
        modelname.head_transformer.d_model=${modelname_head_transformer_d_model} \
        modelname.head_transformer.dim_feedforward=${modelname_head_transformer_dim_feedforward} \
        modelname.head_transformer.dropout=${modelname_head_transformer_dropout} \
        modelname.head_transformer.nhead=${modelname_head_transformer_nhead} \
        modelname.head_transformer.num_layers=${modelname_head_transformer_num_layers} \
        modelname.mmp.attn_steps.dropout=${modelname_mmp_attn_steps_dropout} \
        modelname.mmp.attn_steps.nhead=${modelname_mmp_attn_steps_nhead} \
        modelname.mmp.loss_alignment_alpha=${modelname_mmp_loss_alignment_alpha} \
        modelname.mmp.num_aggregated_tokens=${modelname_mmp_num_aggregated_tokens} \
        modelname.mmp.proj_mlp.dropout=${modelname_mmp_proj_mlp_dropout} \
        modelname.mmp.proj_mlp.hidden_dim=${modelname_mmp_proj_mlp_hidden_dim}