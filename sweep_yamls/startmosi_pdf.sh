#!/usr/bin/env bash
for i in {1..2}; do
    tmux new-session -d -s "pdf_mosi_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/03gqi452"
done