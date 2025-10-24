#!/usr/bin/env bash
for i in {1..2}; do
    tmux new-session -d -s "ogm_ge_mosei_$i" \
        "cd $(pwd); conda run -p /sc-projects/sc-proj-ukb-cvd/environments/mml wandb agent --count 25 cardiors/simple_mml_baseline/pag8awvo"
done