#!/bin/zsh
# Horizon x exogenous-window 2x2.  h=6 first: h=1 is saturated, and its control
# alone varies by more than that horizon's entire headroom.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=3
exec nice -n 10 python3 -u -m experiments.run_spatial_2x2 \
    --horizons 6,1 --seeds 1,2 --threads 3 \
    --out results/spatial_2x2_results.json
