#!/usr/bin/env bash
# Bring PACE results home. scratch is wiped between semesters, so anything that
# only lives there is not a result yet.
set -euo pipefail
cd "$(dirname "$0")/.."
H="${1:-login-phoenix.pace.gatech.edu}"
R="scratch/sa_market"
mkdir -p results/pace preds/pace logs/pace
rsync -az --info=stats1 "$H:$R/results/h6_*.json" results/pace/ 2>/dev/null || true
rsync -az --info=stats1 "$H:$R/preds/h6/" preds/pace/ 2>/dev/null || true
rsync -az "$H:$R/logs/h6_*.out" logs/pace/ 2>/dev/null || true
echo "results/pace: $(ls results/pace/ 2>/dev/null | wc -l) files"
echo "preds/pace:   $(ls preds/pace/ 2>/dev/null | wc -l) files"
