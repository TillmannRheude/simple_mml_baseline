#!/usr/bin/env bash
for i in {1..2}; do
    tmux new-session -d -s "mmp_symile_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 50 cardiors/simple_mml_baseline/ao83n59i"
done