#!/usr/bin/env bash
# Horizon 24 steps = 12 hours. At h=1 the predictable part of this series is a
# lag-1/lag-2 mean reversion worth 2.15 MAE, which 3 parameters capture and a
# band decomposition can only smear: AR(2) scores 27.23 against AR(12)'s 27.22.
# At h=24 persistence collapses from 29.38 to 72.99, a week of history starts
# paying (AR(336) beats AR(96) by 1.31, against -0.05 at h=1) and the daily
# cycle dominates -- the conditions a multi-scale decomposition is for.
set -u
cd "$(dirname "$0")/.."
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv \
        --train-year 2018,2019,2020 --test-year 2021 --horizon 24 \
        --target-transform asinh --loss l1 --seeds 1 --epochs 8 --patience 8 \
        --threads 3 --cache-prefix cache_unf/"
OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --arms nvmd_st,nvmd_temporal,fixed_geo \
  --save-preds preds/h24_2021 --out results/h24_2021.json
