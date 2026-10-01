#!/usr/bin/env bash
# Level target against differenced target, everything else identical.
#
# In the calm 90% of 2021, persistence scores 10.72 against 12.21 for AR(48)
# and 13.8 for a level-trained network: the price is close to a random walk
# there, so any model that moves pays for it. On a differenced target a zero
# output is persistence, and the model only spends where it has something.
set -u
cd "$(dirname "$0")/.."
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv \
        --train-year 2018,2019,2020 --test-year 2021 \
        --target-transform asinh --loss l1 --seeds 1 --epochs 8 --patience 8 \
        --threads 3 --cache-prefix cache_unf/ --arms nvmd_st,nvmd_temporal"
OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --target-mode delta --save-preds preds/delta2021 --out results/delta_2021.json
