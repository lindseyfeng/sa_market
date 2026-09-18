#!/bin/zsh
# Decomposition families and delivery paths, then the churn dose-response.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3
nice -n 12 python3 -u -m experiments.run_three_arms \
    --arms vmd_price_res,fixed_vmdmean,fixed_geo,nvmd_trained,ewt,emd,wpt,bank \
    --seeds 1,2,3 --threads 3 --out results/stability_results.json
echo "=== zoo done $(date); starting dose ==="
exec nice -n 12 python3 -u -m experiments.run_three_arms \
    --arms bank,bankjit0.015,bankjit0.025,bankjit0.04,bankjit0.07,bankjit0.12 \
    --seeds 1,2 --threads 3 --out results/dose_results.json
