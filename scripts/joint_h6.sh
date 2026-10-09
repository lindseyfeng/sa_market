#!/usr/bin/env bash
# The same three arms at h=6, which is where the spatial effect was measured.
#
# FINDINGS section 11a puts the spatial coupling gain at -2.21 MAE at h=6,
# decaying to zero by h=48, and records h=1 as saturated (persistence 14.40
# against a best model near 14.3). The h=1 run of this model came out at -0.1%
# with a DM p of 0.479 -- a null, but a null measured in a window too small to
# hold a 2.21-point effect. h=24 is not the cheaper option: the horizon only
# shifts the target index, so every horizon costs the same per epoch. It is the
# worse option, because the effect is already known to be gone by then.
#
# Only --horizon changes from scripts/joint.sh. Same loss, same arms, same seed,
# same epochs, so the horizon is the one variable.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds
OMP_NUM_THREADS=3 nice -n 5 python3 -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 \
  --horizon 6 \
  --arms "${ARMS:-single:SA1_price,joint,joint_nocouple}" \
  --w-dev "${W_DEV:-1.0}" --w-bias "${W_BIAS:-0.1}" --w-sparse "${W_SPARSE:-1e-4}" \
  --w-aux "${W_AUX:-0.0}" \
  --seeds "${SEEDS:-1}" --epochs "${EPOCHS:-15}" --patience 15 \
  --threads 3 --save-preds preds/joint_h6 --out results/joint_h6.json
