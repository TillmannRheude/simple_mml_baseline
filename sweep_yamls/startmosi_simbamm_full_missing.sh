#!/usr/bin/env bash
for i in {1..4}; do
    tmux new-session -d -s "simbamm_full_missing_mosi_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/bz1pb0ys"
done