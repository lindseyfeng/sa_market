#!/bin/bash
# Full 2018-2022 pipeline on the paper's split (train 2018-2021, test 2022).
# Waits for current jobs so we do not oversubscribe 8 cores further.
set -u
cd /Users/lindseyf/sa_market

while pgrep -f "train_nvmd_v3|benchmark_seeds|generate_modes" >/dev/null; do sleep 30; done
echo "=== CORES FREE, STARTING LONG-DATA PIPELINE ==="

python3 - <<'PY'
import pandas as pd
f = pd.read_csv('SA_filtered_2018_2022.csv', parse_dates=['SETTLEMENTDATE'])
tr = f[f.SETTLEMENTDATE.dt.year <= 2021]
te = f[f.SETTLEMENTDATE.dt.year == 2022]
for d, n in ((tr, 'SA_train_2018_2021.csv'), (te, 'SA_test_2022.csv')):
    d.to_csv(n, index=False)
    print(f'{n}: {len(d)} rows')
PY

echo ""
echo "=== CAUSAL VMD W=96 OVER FULL SERIES ==="
python3 generate_modes.py causal-vmd --csv SA_filtered_2018_2022.csv \
  --output causal_vmd_full_w96.csv --K 12 --window 96 --n-jobs 7 || exit 1

echo ""
echo "=== TRAIN NVMD v3 ON 2018-2021 ==="
python3 -u train_nvmd_v3.py --K 8 --seq-len 96 --epochs 10 --patience 4 \
  --adapt 0 --band-lr 3e-4 \
  --train-csv SA_train_2018_2021.csv --val-csv SA_test_2022.csv \
  --outdir ./runs_v3_long > v3_long.log 2>&1 || exit 1
grep -E "Saved best" v3_long.log

echo ""
echo "=== GENERATE NVMD MODES OVER FULL SERIES ==="
python3 generate_modes.py nvmd --model runs_v3_long/best.pt \
  --csv SA_filtered_2018_2022.csv --output nvmd_full_w96.csv --edge-pad 0 || exit 1

echo ""
echo "=== CHRONOLOGICAL SPLIT AT 2022-01-01 ==="
python3 - <<'PY'
import pandas as pd
for src, tag in (('causal_vmd_full_w96.csv', 'cvmd'), ('nvmd_full_w96.csv', 'nvmd')):
    d = pd.read_csv(src, parse_dates=['SETTLEMENTDATE'])
    tr = d[d.SETTLEMENTDATE.dt.year <= 2021]
    te = d[d.SETTLEMENTDATE.dt.year == 2022]
    tr.to_csv(f'{tag}_long_train.csv', index=False)
    te.to_csv(f'{tag}_long_test.csv', index=False)
    print(f'{tag}: train={len(tr)} test={len(te)}')
PY

echo ""
echo "=== AR LEAKAGE GATE (long split) ==="
python3 ar_probe.py \
  --method "Causal VMD" cvmd_long_train.csv cvmd_long_test.csv \
  --method "NVMD v3"    nvmd_long_train.csv nvmd_long_test.csv

echo ""
echo "=== INTERPRETABILITY (long split) ==="
python3 interpret_modes.py \
  --method "Causal VMD" cvmd_long_train.csv cvmd_long_test.csv \
  --method "NVMD v3"    nvmd_long_train.csv nvmd_long_test.csv

echo ""
echo "=== BENCHMARK (paper split: train 2018-2021, test 2022) ==="
python3 benchmark_seeds.py \
  --method "Causal VMD" cvmd_long_train.csv cvmd_long_test.csv \
  --method "NVMD v3"    nvmd_long_train.csv nvmd_long_test.csv \
  --ref "Causal VMD" --seeds 5 --window 48 --epochs 10 --patience 4 --skip-cnn-bilstm

echo "=== LONGDATA COMPLETE ==="
