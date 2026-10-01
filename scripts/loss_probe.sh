#!/usr/bin/env bash
# Is the 1.56 MAE nvmd_st gives away on the calm 90% a loss-function artefact?
# Huber beta=1 acts on standardised asinh values, where nearly every sample is
# inside the quadratic region, so the tail is clipped twice: once by asinh and
# again by Huber. The metric is MAE, so train L1 against it directly.
set -u
cd "$(dirname "$0")/.."
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv \
        --train-year 2018,2019,2020 --test-year 2021 \
        --target-transform asinh --seeds 1 --epochs 8 --patience 8 \
        --threads 3 --cache-prefix cache_unf/ --arms nvmd_st --bias-correct"
for cfg in "l1:"; do
  loss=${cfg%%:*}; extra=${cfg#*:}
  OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
    --loss "$loss" $extra --save-preds "preds/probe_${loss}${extra// /}" \
    --out "results/probe_${loss}${extra// /}.json"
done
