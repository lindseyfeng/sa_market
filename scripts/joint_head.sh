#!/usr/bin/env bash
# The defining test of whether the decomposition is worth anything.
#
# A ridge on the raw 96 x 37 window beat the LSTM-head network by 1.3% at h=6
# (37.097 against 37.575, p=0.040). The LSTM hands its *last hidden state* to an
# MLP, so 96 steps are compressed into one vector, while the ridge reads every
# one of the 3,552 values linearly. That confounds representation with function
# class: the band stack was never given a reader that could do what the ridge
# does.
#
# A linear head over the whole (2K, L) stack removes the confound. Then
#
#     joint/linear   linear on 96 x 16 band features
#     arx_window     ridge on 96 x 37 raw features   (done: 37.097)
#
# are the same function class on the same information, and the only difference
# is the representation. If the bands win there, the decomposition earned its
# place; if they lose, no head will rescue it.
#
#   joint/linear            bands + coupling, linear reader
#   joint_nocouple/linear   bands without coupling: is the coupling worth
#                           anything once the reader is not the bottleneck
#
# The linear head has 70,750 parameters against the LSTM's 648,281 and no
# recurrence, so this costs a fraction of the earlier runs -- which matters on a
# box that drove one epoch to 23,560 s under swap pressure.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds
OMP_NUM_THREADS=3 nice -n 5 python3 -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 --horizon 6 \
  --arms "${ARMS:-joint/linear,joint_nocouple/linear}" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 \
  --seeds 1 --epochs "${EPOCHS:-15}" --patience 15 \
  --threads 3 --save-preds preds/joint_head --out results/joint_head.json
