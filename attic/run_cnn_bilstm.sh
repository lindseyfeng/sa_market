#!/bin/bash
# Fills attic/RESULTS-superseded.md section 7, item 2: CNN-BiLSTM is absent from every multi-seed
# table, and it is the closest analogue to the published baseline -- plus the one
# cell where NVMD lost on a single seed (18.75 vs 17.36).  Its absence reads as
# selection, so it has to be measured whichever way it lands.
#
# --only CNN-BiLSTM (added to benchmark_seeds.py) runs just the missing cell
# instead of re-deriving Linear/MLP/LSTM, which are already tabulated.
# 5 seeds to match the section 8 protocol.
set -u
cd /Users/lindseyf/sa_market

while pgrep -f "run_vmd_screen.sh" >/dev/null; do sleep 60; done
echo "=== screen finished, starting CNN-BiLSTM ==="

python3 -u benchmark_seeds.py \
  --method "Causal VMD"    causal_vmd_2018_2018_w96.csv  causal_vmd_2019_2019_w96.csv \
  --method "NVMD v2"       nvmd_modes_2018_2018_pad0.csv nvmd_modes_2019_2019_pad0.csv \
  --method "NVMD v3static" v3static_modes_2018_2018.csv  v3static_modes_2019_2019.csv \
  --ref "Causal VMD" --seeds 5 --window 96 --horizon 1 \
  --epochs 30 --patience 7 --only CNN-BiLSTM
echo "=== CNN-BILSTM COMPLETE ==="
