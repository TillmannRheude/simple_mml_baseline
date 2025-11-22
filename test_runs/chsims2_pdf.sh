#!/bin/bash

#SBATCH --job-name=SB-CV-PDF
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

modelname_optimizer_lr=0.00026300730620861505
modelname_optimizer_warmup_steps=100
modelname_optimizer_weight_decay=0.1
modelname_pdf_loss_weight=0.020729606356623156
modelname_pdf_p_head_dropout=0
modelname_pdf_p_head_hidden_dims="[256]"
modelname_pdf_unimodal_loss_weight=0.044674399886194455

batch_size=8

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="ch_sims_v2" \
        encoders="ch_sims_v2" \
        modelname=pdf \
        wandb.group="CV-CHSims2-PDF" \
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
        modelname.pdf.loss_weight=${modelname_pdf_loss_weight} \
        modelname.pdf.p_head.dropout=${modelname_pdf_p_head_dropout} \
        modelname.pdf.p_head.hidden_dims=${modelname_pdf_p_head_hidden_dims} \
        modelname.pdf.unimodal_loss_weight=${modelname_pdf_unimodal_loss_weight}