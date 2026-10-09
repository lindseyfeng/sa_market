#!/usr/bin/env bash
# Joint multi-region prediction, with the loss the detectability check argued for.
#
# Arm order matters: the headline comparison (SA1 single-task vs joint, and
# whether it is the coupling or the multi-task effect) lands first, because the
# runner dumps atomically after each arm and the whole queue is many hours.
#
# nice and a thread cap are not optional on this box: without them epoch times
# swing from 91 s to 6451 s under swap pressure.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds
OMP_NUM_THREADS=3 nice -n 5 python3 -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 \
  --arms "${ARMS:-single:SA1_price,joint,joint_nocouple,single:NSW1_price,single:VIC1_price,single:QLD1_price,single:TAS1_price}" \
  --w-dev "${W_DEV:-1.0}" --w-bias "${W_BIAS:-0.1}" --w-sparse "${W_SPARSE:-1e-4}" \
  --w-aux "${W_AUX:-0.0}" \
  --seeds "${SEEDS:-1}" --epochs "${EPOCHS:-15}" --patience 15 \
  --threads 3 --save-preds preds/joint --out results/joint.json
