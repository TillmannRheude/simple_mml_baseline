#!/bin/bash

#SBATCH --job-name=SB-CV
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

encoders_audio_transformer_dropout=0.1
encoders_audio_transformer_intermediate_size=512
encoders_audio_transformer_num_attention_heads=4
encoders_audio_transformer_num_hidden_layers=8
encoders_vision_transformer_dropout=0.1
encoders_vision_transformer_intermediate_size=256
encoders_vision_transformer_num_attention_heads=1
encoders_vision_transformer_num_hidden_layers=8
modelname_head_transformer_d_model=256
modelname_head_transformer_dim_feedforward=2048
modelname_head_transformer_dropout=0.0
modelname_head_transformer_nhead=16
modelname_head_transformer_num_layers=6

batch_size=128
modelname_optimizer_lr=0.01039262404706731
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0.01
modelname_bmml_alpha=0.0026625036732789956
modelname_bmml_bmml_momentum=0.99
modelname_bmml_q=5
modelname_bmml_unimodal_loss_weight=0.21441205182007755
modelname_bmml_warmup_epochs=10


python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="mosi" \
        encoders="mosi" \
        modelname=bmml \
        wandb.group="CV-MOSI-SimBaMM-BMML" \
        missing.missing_train=[0.0,0.0] \
        missing.missing_valid=[0.0,0.0] \
        missing.missing_test=[0.0,0.0] \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        batch_size=${batch_size} \
        encoders.audio.transformer.dropout=${encoders_audio_transformer_dropout} \
        encoders.audio.transformer.intermediate_size=${encoders_audio_transformer_intermediate_size} \
        encoders.audio.transformer.num_attention_heads=${encoders_audio_transformer_num_attention_heads} \
        encoders.audio.transformer.num_hidden_layers=${encoders_audio_transformer_num_hidden_layers} \
        encoders.vision.transformer.dropout=${encoders_vision_transformer_dropout} \
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