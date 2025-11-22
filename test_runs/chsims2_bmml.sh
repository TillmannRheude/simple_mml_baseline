#!/bin/bash

#SBATCH --job-name=SB-CV-BMML
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

# modelname.optimizer.lr=9.7678379942058e-05 modelname.optimizer.warmup_steps=200 modelname.optimizer.weight_decay=0.01 modelname.bmml.alpha=0.08933948652574326 modelname.bmml.bmml_momentum=0.9 modelname.bmml.q=3 modelname.bmml.unimodal_loss_weight=0.03022424798599897 modelname.bmml.warmup_epochs=1

modelname_optimizer_lr=9.7678379942058e-05
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0.01
modelname_bmml_alpha=0.08933948652574326
modelname_bmml_bmml_momentum=0.9
modelname_bmml_q=3
modelname_bmml_unimodal_loss_weight=0.03022424798599897
modelname_bmml_warmup_epochs=1

batch_size=8

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="ch_sims_v2" \
        encoders="ch_sims_v2" \
        modelname=bmml \
        wandb.group="CV-CHSims2-BMML" \
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
        modelname.bmml.alpha=${modelname_bmml_alpha} \
        modelname.bmml.bmml_momentum=${modelname_bmml_bmml_momentum} \
        modelname.bmml.q=${modelname_bmml_q} \
        modelname.bmml.unimodal_loss_weight=${modelname_bmml_unimodal_loss_weight} \
        modelname.bmml.warmup_epochs=${modelname_bmml_warmup_epochs}