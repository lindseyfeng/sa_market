#!/usr/bin/env bash
# Does seeding the coupling off zero close the gap to the ridge? One arm, because
# the control already exists.
#
# The finished h=6 `joint` run is 15 epochs with the coupling at exactly zero, so
# `joint@0.01` at 15 epochs with everything else identical is the controlled
# contrast for the initialisation on its own. The earlier plan ran 25 epochs of
# both and ordered zero-init first; it thrashed the box at epoch 8 -- 449 s per
# epoch rising to 23,560 s -- and never reached the arm that mattered.
#
#   joint        (done)  zero init, 15 epochs, SA1 37.575
#   joint@0.01           seeded init, 15 epochs
#   arx_window   (done)  ridge on the same window, SA1 37.097  <- the bar
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds
OMP_NUM_THREADS=3 nice -n 5 python3 -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 --horizon 6 \
  --arms "${ARMS:-joint@0.01}" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 \
  --seeds 1 --epochs "${EPOCHS:-15}" --patience 15 \
  --threads 3 --save-preds preds/joint_init --out results/joint_init.json
