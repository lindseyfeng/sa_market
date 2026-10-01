#!/usr/bin/env bash
# The original protocol -- train 2018, test 2019, h=1 -- on the unfiltered
# series, so negative prices and the spikes above 981.65 are back in. One
# training year keeps an epoch near a minute.
set -u
cd "$(dirname "$0")/.."
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv \
        --train-year 2018 --test-year 2019 --horizon 1 \
        --target-transform asinh --loss l1 --seeds 1 --epochs 10 --patience 10 \
        --threads 3 --cache-prefix cache_unf/ --modes cache_unf/vmd_panel_K8_a1000_W96 \
        --decomp-channels SA1_price"
OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --arms nvmd_st,nvmd_temporal,fixed_geo,vmd_price_res \
  --save-preds preds/fast2019 --out results/fast_2019.json
