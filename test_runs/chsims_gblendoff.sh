#!/bin/bash

#SBATCH --job-name=SB-CV-GblendOff
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

encoders_audio_transformer_num_hidden_layers=4
encoders_audio_transformer_num_attention_heads=4
encoders_audio_transformer_intermediate_size=1024
encoders_vision_transformer_num_hidden_layers=4
encoders_vision_transformer_num_attention_heads=2
encoders_vision_transformer_intermediate_size=512
modelname_head_transformer_d_model=128
modelname_head_transformer_dim_feedforward=2048
modelname_head_transformer_dropout=0
modelname_head_transformer_nhead=16
modelname_head_transformer_num_layers=6

batch_size=16
# modelname.optimizer.lr=2.7730341856626473e-05 modelname.optimizer.warmup_steps=0 modelname.optimizer.weight_decay=0.1 modelname.gblend.lookahead_epochs=1 modelname.gblend.mode=offline modelname.gblend.update_freq=5
modelname_optimizer_lr=2.7730341856626473e-05
modelname_optimizer_warmup_steps=0
modelname_optimizer_weight_decay=0.1
modelname_gblend_lookahead_epochs=1
modelname_gblend_mode=offline
modelname_gblend_update_freq=5

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="ch_sims" \
        encoders="ch_sims" \
        modelname=gblend \
        wandb.group="CV-CHSims-GblendOff" \
        missing.missing_train=[0.0,0.0] \
        missing.missing_valid=[0.0,0.0] \
        missing.missing_test=[0.0,0.0] \
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
        modelname.gblend.lookahead_epochs=${modelname_gblend_lookahead_epochs} \
        modelname.gblend.mode=${modelname_gblend_mode} \
        modelname.gblend.update_freq=${modelname_gblend_update_freq} \