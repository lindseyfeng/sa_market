#!/usr/bin/env bash
# Every arm on the unfiltered series, train 2018-2020 / test 2021-2022,
# asinh target. One seed: nothing has been validated under this protocol yet,
# so the first pass is for the landscape, not for deciding between arms.
# CLAUDE.md's seed-noise warning still stands -- add seeds to a pair only once
# that pair looks separated.
set -u
cd "$(dirname "$0")/.."
PANEL=data/raw/compound_unfiltered_2018_2022.csv
COMMON="--panel $PANEL --train-year 2018,2019,2020 --test-year 2021,2022 \
        --target-transform asinh --loss huber --huber-beta 1.0 \
        --seeds 1 --threads 6 --epochs 15 --patience 15 --cache-prefix cache_unf/"

OMP_NUM_THREADS=6 nice -n 10 python3 -m experiments.run_three_arms $COMMON \
  --arms fixed_geo,fixed_vmdmean,nvmd_trained \
  --out results/internal_2021_2022.json

# the zoo arms need their mode caches, which the generator writes last
until [ "$(ls cache_unf/decomp_emd_K8_W96/*.npy 2>/dev/null | wc -l)" -ge 10 ]; do sleep 60; done
OMP_NUM_THREADS=6 nice -n 10 python3 -m experiments.run_three_arms $COMMON \
  --arms bank,wpt,ewt,emd \
  --out results/zoo_2021_2022.json

until [ "$(ls cache_unf/vmd_panel_K8_a1000_W96/*.npy 2>/dev/null | wc -l)" -ge 5 ]; do sleep 60; done
OMP_NUM_THREADS=6 nice -n 10 python3 -m experiments.run_three_arms $COMMON \
  --modes cache_unf/vmd_panel_K8_a1000_W96 --arms vmd_price_res \
  --out results/vmd_2021_2022.json
