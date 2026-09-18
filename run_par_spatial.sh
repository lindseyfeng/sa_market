#!/bin/zsh
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=3
exec nice -n 10 python3 -u run_spatial_2x2.py \
    --horizons 6,1 --seeds 1,2 --threads 3 --out spatial_2x2_results.json
