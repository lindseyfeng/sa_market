#!/usr/bin/env bash
# Six arms on one quarter, to separate four questions at once:
#
#   lstm_panel  vs  nvmd_st        does decomposing help at all?
#   nvmd_st     vs  nvmd_temporal  does the spatial coupling help?
#   lstm        vs  lstm_panel     do the exogenous channels help?
#   nvmd_st+lin vs  AR(96)         does the decomposition add to a linear AR?
#
# Q2 2019 is the quietest quarter (persistence 33.97 against 149.12 in Q1), so
# it has the best signal-to-noise for separating arms. The sliding window is
# the same one the rolling run used for Q2.
set -u
cd "$(dirname "$0")/.."
until ! pgrep -f "run_three_arms" >/dev/null; do sleep 60; done
COMMON="--panel data/raw/compound_unfiltered_2018_2022.csv --horizon 24 \
        --train-year 2018-04-01:2019-03-31 --test-year 2019-04-01:2019-06-30 \
        --target-transform asinh --loss l1 --seeds 1 --epochs 10 --patience 10 \
        --threads 3 --cache-prefix cache_unf/"

OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --arms lstm,lstm_panel,nvmd_temporal,nvmd_st \
  --save-preds preds/abl_q2 --out results/abl_q2.json

OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --arms nvmd_st,lstm_panel --linear-residual \
  --save-preds preds/abl_q2_lin --out results/abl_q2_lin.json
