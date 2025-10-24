#!/usr/bin/env bash
for i in {1..5}; do
    tmux new-session -d -s "lmf_haim_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 20 cardiors/simple_mml_baseline/160wxw8y"
done