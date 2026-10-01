#!/usr/bin/env bash
# Rolling expanding-window retrain, which is what recalibration looks like in
# practice and what the EPF literature evaluates on.
#
# Training 2018-2020 and testing 2021-2022 asks the model to extrapolate into
# a regime the training period contains nothing like: persistence is 23.65 on
# the training years, 29.38 on 2021 and 70.35 on 2022, with 2022-07 at 133.7.
# Each fold below refits on everything before its test period, so 2022 is
# forecast by a model that has seen 2021.
#
#   fold 1  train 2018-2020       test 2021
#   fold 2  train 2018-2021       test 2022
set -u
cd "$(dirname "$0")/.."
PANEL=data/raw/compound_unfiltered_2018_2022.csv
ARMS=${ARMS:-nvmd_st,nvmd_temporal,nvmd_trained,fixed_geo,fixed_vmdmean,vmd_price_res,vmd_price,bank,wpt,ewt,emd}
COMMON="--panel $PANEL --target-transform asinh --loss huber --huber-beta 1.0 \
        --seeds 1 --epochs 15 --patience 15 --threads 3 --cache-prefix cache_unf/ \
        --modes cache_unf/vmd_panel_K8_a1000_W96 --decomp-channels SA1_price"

OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --train-year 2018,2019,2020 --test-year 2021 --arms "$ARMS" \
  --save-preds preds/fold2021 --out results/rolling_2021.json

OMP_NUM_THREADS=3 nice -n 5 python3 -m experiments.run_three_arms $COMMON \
  --train-year 2018,2019,2020,2021 --test-year 2022 --arms "$ARMS" \
  --save-preds preds/fold2022 --out results/rolling_2022.json
