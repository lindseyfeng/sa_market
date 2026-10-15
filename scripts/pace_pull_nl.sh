#!/usr/bin/env bash
# Bring the nonlinear sweep home. Anything that only lives on scratch is not a
# result yet -- scratch is wiped between semesters.
#
# Predictions and checkpoints only; the logs stay on the cluster. The .npz
# files are what every paired test in analysis/ reads, and the .pt files carry
# gate_c, the band centres and the coupling, which is the difference between
# "this arm scored 37.5" and "this arm scored 37.5 and here is what it learned".
set -euo pipefail
# macOS ships rsync 2.6.9: no --info=, no -z on some builds. Keep the flags boring.
cd "$(dirname "$0")/.."
H="${HORIZON:-6}"
R="login-phoenix.pace.gatech.edu"
S="-o ControlMaster=no -o ControlPath=none -o ConnectTimeout=25"

mkdir -p "preds/nl${H}" "runs/nl${H}" logs/pace
# ~ is not expanded remotely inside a quoted rsync path; use a home-relative one.
rsync -az -e "ssh $S" \
  "$R:scratch/sa_market/preds/nl${H}/" "preds/nl${H}/"
rsync -az -e "ssh $S" \
  "$R:scratch/sa_market/runs/nl${H}/" "runs/nl${H}/"
rsync -az -e "ssh $S" \
  --include="nl${H}_*" --exclude="*" \
  "$R:scratch/sa_market/results/" "results/"
rsync -az -e "ssh $S" \
  --include="nl${H}_*.out" --exclude="*" \
  "$R:scratch/sa_market/logs/" "logs/pace/"

echo
find "preds/nl${H}" -name "*.npz" | sort
find "runs/nl${H}" -name "*.pt" | sort
