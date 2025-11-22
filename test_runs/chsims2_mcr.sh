#!/bin/bash

#SBATCH --job-name=SB-CV-MCR
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

modelname_optimizer_lr=6.806742076969291e-06
modelname_optimizer_warmup_steps=0
modelname_optimizer_weight_decay=0.001
modelname_mcr_ceb_reconstruction_head_hidden_dim=32
modelname_mcr_ceb_reconstruction_head_num_layers=3
modelname_mcr_contrastive_temp=0.028601283071147267
modelname_mcr_loss_weights_ceb=0.9992159727354047
modelname_mcr_loss_weights_con=0.26854120266495873
modelname_mcr_loss_weights_mipd=0.1630961038183308
modelname_mcr_loss_weights_uni="[0.3,0.3,0.3]"
modelname_mcr_num_permutations=10
modelname_mcr_strategy=Greedy

batch_size=8

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="ch_sims_v2" \
        encoders="ch_sims_v2" \
        modelname=mcr \
        wandb.group="CV-CHSims2-MCR" \
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
        modelname.mcr.ceb_reconstruction_head.hidden_dim=${modelname_mcr_ceb_reconstruction_head_hidden_dim} \
        modelname.mcr.ceb_reconstruction_head.num_layers=${modelname_mcr_ceb_reconstruction_head_num_layers} \
        modelname.mcr.contrastive_temp=${modelname_mcr_contrastive_temp} \
        modelname.mcr.loss_weights.ceb=${modelname_mcr_loss_weights_ceb} \
        modelname.mcr.loss_weights.con=${modelname_mcr_loss_weights_con} \
        modelname.mcr.loss_weights.mipd=${modelname_mcr_loss_weights_mipd} \
        modelname.mcr.loss_weights.uni=${modelname_mcr_loss_weights_uni} \
        modelname.mcr.num_permutations=${modelname_mcr_num_permutations} \
        modelname.mcr.strategy=${modelname_mcr_strategy}