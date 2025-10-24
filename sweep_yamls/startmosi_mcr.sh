#!/usr/bin/env bash
for i in {1..3}; do
    tmux new-session -d -s "mcr_mosi_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/wl8agy60"
done