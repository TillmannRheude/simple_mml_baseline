#!/bin/bash

#SBATCH --job-name=SB-CV-CREMAD-BMML
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}

modelname_head_transformer_d_model=64
modelname_head_transformer_dim_feedforward=1024
modelname_head_transformer_dropout=0.1
modelname_head_transformer_nhead=4
modelname_head_transformer_num_layers=4

# modelname.optimizer.lr=0.007499035661351397 modelname.optimizer.warmup_steps=100 modelname.optimizer.weight_decay=0.001 modelname.bmml.alpha=0.0013339548851851843 modelname.bmml.bmml_momentum=0.999 modelname.bmml.q=3 modelname.bmml.unimodal_loss_weight=0.06898925125626548 modelname.bmml.warmup_epochs=10

modelname_optimizer_lr=0.007499035661351397
modelname_optimizer_warmup_steps=100
modelname_optimizer_weight_decay=0.001
modelname_bmml_alpha=0.0013339548851851843
modelname_bmml_bmml_momentum=0.999
modelname_bmml_q=3
modelname_bmml_unimodal_loss_weight=0.06898925125626548
modelname_bmml_warmup_epochs=10

batch_size=128

python3 /sc-projects/sc-proj-ukb-cvd/projects/simple_mml_baseline_tr/main_incltest.py split_nr=${split_nr} \
        dataset="crema_d" \
        encoders="crema_d" \
        modelname=bmml \
        wandb.group="CV-CREMAD-BMML" \
        missing.missing_train=[0.0] \
        missing.missing_valid=[0.0] \
        missing.missing_test=[0.0] \
        modelname.optimizer.lr=${modelname_optimizer_lr} \
        modelname.optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        modelname.optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        batch_size=${batch_size} \
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