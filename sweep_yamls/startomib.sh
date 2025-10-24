#!/usr/bin/env bash
for i in {2..4}; do
    tmux new-session -d -s "omib_haim_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 20 cardiors/simple_mml_baseline/llga3u2t"
done