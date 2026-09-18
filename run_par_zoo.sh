#!/bin/zsh
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=3
nice -n 12 python3 -u run_three_arms.py \
    --arms vmd_price_res,fixed_vmdmean,fixed_geo,nvmd_trained,ewt,emd,wpt,bank \
    --seeds 1,2,3 --threads 3 --out stability_results.json
echo "=== zoo done $(date); starting dose ==="
exec nice -n 12 python3 -u run_three_arms.py \
    --arms bank,bankjit0.015,bankjit0.025,bankjit0.04,bankjit0.07,bankjit0.12 \
    --seeds 1,2 --threads 3 --out dose_results.json
