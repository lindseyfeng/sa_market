#!/usr/bin/env bash
# Sliding one-year window, refit each quarter, h=24 (12 hours ahead).
#
# The training window stays one year throughout, so capacity and sample size
# are held fixed and the only thing that changes is how recent the data is.
# AR is fit once on 2018 and never refit, which is the contrast: does the
# network need recalibration that a linear model does not.
#
#   Q1 2019  train 2018-01..2018-12
#   Q2 2019  train 2018-04..2019-03
#   Q3 2019  train 2018-07..2019-06
#   Q4 2019  train 2018-10..2019-09
set -u
cd "$(dirname "$0")/.."
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv --horizon 24 \
        --target-transform asinh --loss l1 --seeds 1 --epochs 10 --patience 10 \
        --threads 3 --cache-prefix cache_unf/ --arms nvmd_st,nvmd_temporal"

run () {   # $1 train range, $2 test range, $3 tag
  OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
    --train-year "$1" --test-year "$2" \
    --save-preds "preds/roll_$3" --out "results/roll_$3.json"
}
run 2018-01-01:2018-12-31  2019-01-01:2019-03-31  q1
run 2018-04-01:2019-03-31  2019-04-01:2019-06-30  q2
run 2018-07-01:2019-06-30  2019-07-01:2019-09-30  q3
run 2018-10-01:2019-09-30  2019-10-01:2019-12-31  q4
