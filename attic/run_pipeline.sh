#!/bin/zsh
# Four-arm comparison, both selection rules recorded per run.
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=6
exec nice -n 10 python3 -u run_three_arms.py \
    --arms vmd_price,nvmd_st,vmd_panel,nvmd_temporal \
    --seeds 1,2,3 \
    --threads 6 \
    --out three_arms_results.json
