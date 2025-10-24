#!/usr/bin/env bash
for i in {1..4}; do
    tmux new-session -d -s "arl_mosi_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/9b5vygqk"
done