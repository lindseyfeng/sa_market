#!/usr/bin/env bash
# The baseline NVMD has to beat. Waits for the causal VMD mode cache, which the
# generator is still writing for the two test years.
set -u
cd "$(dirname "$0")/.."
until [ "$(ls cache_unf/vmd_panel_K8_a1000_W96/*.npy 2>/dev/null | wc -l)" -ge 5 ]; do sleep 60; done
OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms --panel data/raw/compound_unfiltered_2018_2022.csv --train-year 2018,2019,2020 --test-year 2021,2022 --target-transform asinh --loss huber --huber-beta 1.0 --seeds 1 --epochs 15 --patience 15 --cache-prefix cache_unf/ \
  --threads 3 --modes cache_unf/vmd_panel_K8_a1000_W96 --decomp-channels SA1_price \
  --arms vmd_price,vmd_price_res --out results/vmd_2021_2022.json
