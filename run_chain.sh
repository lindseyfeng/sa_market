#!/bin/zsh
# Spatial 2x2 first -- it is the experiment that was actually asked for, and it
# answers whether the spatial panel is useless or was tested at a saturated
# horizon on trailing exogenous values.  Zoo and dose resume afterwards; both
# skip the runs already in their result files.
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=6

nice -n 10 python3 -u run_spatial_2x2.py \
    --horizons 1,6 --seeds 1,2 --threads 6 --out spatial_2x2_results.json

echo "=== spatial done $(date); resuming zoo ==="
nice -n 10 python3 -u run_three_arms.py \
    --arms vmd_price_res,fixed_vmdmean,fixed_geo,nvmd_trained,ewt,emd,wpt,bank \
    --seeds 1,2,3 --threads 6 --out stability_results.json

echo "=== zoo done $(date); starting dose ==="
exec nice -n 10 python3 -u run_three_arms.py \
    --arms bank,bankjit0.015,bankjit0.025,bankjit0.04,bankjit0.07,bankjit0.12 \
    --seeds 1,2 --threads 6 --out dose_results.json
