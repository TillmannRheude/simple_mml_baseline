#!/usr/bin/env bash
for i in {1..2}; do
    tmux new-session -d -s "gblendon_haim_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/j8z91ikt"
done