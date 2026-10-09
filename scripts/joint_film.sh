#!/usr/bin/env bash
# The two changes that could let the decomposition beat a ridge, kept separable.
#
# Why not a linear residual branch: initialising at the ridge solution would
# guarantee the headline number but would prove nothing about the bands -- the
# ridge would be doing the work. Rejected deliberately.
#
# Why these two. A partition-of-unity decomposition is an invertible linear map
# and the per-band coupling is a weighted sum, so the whole additive pathway
# sits inside the span of a linear map on the raw window: ridge on bands and
# ridge on the raw window return identical predictions to 0.0000 $/MWh,
# measured. Nothing additive can beat a ridge, however it is read.
#
#   +film     CrossFilter's FiLM gate multiplies the own bands by a function of
#             the exogenous state. A product is outside the span of any linear
#             map, and it is the point of having bands at all: the gate says
#             which time scale the exogenous state modulates.
#   +ctx336   the bank sees 336 steps, the head still reads 96. At window 96
#             nothing above 20.1 h exists in the bank's output, so the 168 h
#             weekly cycle -- the period the field's naive benchmark is built on
#             -- is absent from the input rather than merely unused.
#
#   film effect     joint/lstm+film  vs  the finished joint (37.575)
#   context effect  joint/lstm+film+ctx336  vs  joint/lstm+film
#
# Both arms share one window list, raised to skip_head 240 so the 336-step
# context fits; eval_joint intersects rows against the 96-step runs.
# The bar is arx_window at 37.097.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds
OMP_NUM_THREADS=3 nice -n 5 python3 -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 --horizon 6 \
  --arms "${ARMS:-joint/lstm+film,joint/lstm+film+ctx336}" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 \
  --seeds 1 --epochs "${EPOCHS:-15}" --patience 15 \
  --threads 3 --save-preds preds/joint_film --out results/joint_film.json
