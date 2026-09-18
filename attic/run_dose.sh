#!/bin/zsh
# 1. dose-response: is basis churn causal, or only correlated with error?
# 2. spatial 2x2: was the spatial panel useless, or tested where there was no
#    headroom and fed only trailing exogenous values?
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=6

nice -n 10 python3 -u run_three_arms.py \
    --arms bank,bankjit0.015,bankjit0.025,bankjit0.04,bankjit0.07,bankjit0.12 \
    --seeds 1,2 \
    --threads 6 \
    --out dose_results.json

echo "=== dose done $(date); starting spatial 2x2 ==="
exec nice -n 10 python3 -u run_spatial_2x2.py \
    --horizons 1,6 \
    --seeds 1,2 \
    --threads 6 \
    --out spatial_2x2_results.json
