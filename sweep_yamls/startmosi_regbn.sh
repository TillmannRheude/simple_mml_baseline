#!/usr/bin/env bash
for i in {1..3}; do
    tmux new-session -d -s "regbn_mosi_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/n6ezat4v"
done