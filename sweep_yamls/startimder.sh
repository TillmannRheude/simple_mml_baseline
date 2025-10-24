#!/usr/bin/env bash
for i in {1..2}; do
    tmux new-session -d -s "imder_haim_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 20 cardiors/simple_mml_baseline/2431sir1"
done