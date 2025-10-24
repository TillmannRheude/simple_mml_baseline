#!/usr/bin/env bash
for i in {1..4}; do
    tmux new-session -d -s "dgl_chsims_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml_rocm wandb agent --count 25 cardiors/simple_mml_baseline/an7srq20"
done