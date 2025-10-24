#!/usr/bin/env bash
for i in {1..1}; do
    tmux new-session -d -s "mcr_haim_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml_rocm wandb agent --count 75 cardiors/simple_mml_baseline/25m68jdj"
done