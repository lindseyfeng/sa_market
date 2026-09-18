#!/bin/zsh
# Why does a learned bank beat VMD?  Two hypotheses, separated:
#
#   allocation  -- NVMD puts 5 of 8 bands below f=0.08 where the price energy
#                  is; VMD spreads them near-uniformly
#   stability   -- NVMD's bank is identical in every window; VMD re-solves it,
#                  and its centres move 12.4% of a band gap between adjacent
#                  windows, crossing half a gap in 8.2% of steps
#
# fixed_vmdmean is the decisive cell: VMD's band layout, held fixed.  If it
# matches nvmd_trained, allocation is not the mechanism and learning is not
# either -- only stability is.  The zoo arms then test whether that scalar
# predicts accuracy across decomposition families.
cd /Users/lindseyf/sa_market
export OMP_NUM_THREADS=6
exec nice -n 10 python3 -u run_three_arms.py \
    --arms vmd_price_res,fixed_vmdmean,fixed_geo,nvmd_trained,ewt,emd,wpt,bank \
    --seeds 1,2,3 \
    --threads 6 \
    --out stability_results.json
