#!/usr/bin/env bash
# The method itself, in priority order. nvmd_st carries the spatial claims and
# nvmd_temporal is its matched control, so they go first; the fixed-bank arms
# are the within-family controls and follow.
set -u
cd "$(dirname "$0")/.."
OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms --panel data/raw/compound_unfiltered_2018_2022.csv --train-year 2018,2019,2020 --test-year 2021,2022 --target-transform asinh --loss huber --huber-beta 1.0 --seeds 1 --epochs 15 --patience 15 --cache-prefix cache_unf/ \
  --threads 3 --arms nvmd_st,nvmd_temporal,nvmd_trained,fixed_geo \
  --out results/nvmd_2021_2022.json
