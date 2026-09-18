#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# Causal VMD vs NVMD v3static at h=1 (anchor: should reproduce the published
# ~14.43 / 14.28) and h=6 (the horizon where the exogenous panel pays).
# window 96 to match the spatio-temporal runs; CNN-BiLSTM skipped (5.7M params,
# 10 min/seed, and it was never competitive).
for h in 1 6; do
  python3 -u benchmark_seeds.py \
    --method "Causal VMD"   causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
    --method "NVMD v3static" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv \
    --ref "Causal VMD" --seeds 3 --window 96 --horizon $h \
    --epochs 30 --patience 6 --skip-cnn-bilstm > vmd_h${h}.log 2>&1
  echo "=== h=$h done ==="
  sed -n '/Val MAE, mean/,$p' vmd_h${h}.log
done
echo "=== VMD HORIZON COMPLETE ==="
