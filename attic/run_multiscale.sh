#!/bin/bash
# Wait for the L=192/384 runs, build multi-scale mode CSVs, benchmark them.
set -u
cd /Users/lindseyf/sa_market

while pgrep -f "runs_ms_L" >/dev/null; do sleep 20; done
echo "=== MULTI-SCALE TRAINING DONE ==="
grep -H "Saved best" ms_L192.log ms_L384.log

for L in 192 384; do
  ck="runs_ms_L${L}/best.pt"
  [ -f "$ck" ] || { echo "ABORT: missing $ck"; exit 1; }
  for yr in 2018_2018 2019_2019; do
    python3 generate_modes.py nvmd --model "$ck" \
      --csv "../sa_market2/VMD_modes_with_residual_${yr}.csv" \
      --output "ms_L${L}_modes_${yr}.csv" --edge-pad 0 || exit 1
  done
done

echo ""
echo "=== MERGING SCALES ==="
for yr in 2018_2018 2019_2019; do
  python3 merge_scales.py --out "ms_all_${yr}.csv" \
    "v3static_modes_${yr}.csv" "ms_L192_modes_${yr}.csv" "ms_L384_modes_${yr}.csv" || exit 1
done

echo ""
echo "=== MULTI-SCALE BENCHMARK vs causal VMD ==="
python3 benchmark_seeds.py \
  --method "Causal VMD" causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
  --method "v3 L96" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv \
  --method "v3 L384" ms_L384_modes_2018_2018.csv ms_L384_modes_2019_2019.csv \
  --method "v3 multiscale" ms_all_2018_2018.csv ms_all_2019_2019.csv \
  --ref "Causal VMD" --seeds 5 --window 48 --epochs 10 --patience 4 --skip-cnn-bilstm

echo "=== MULTISCALE COMPLETE ==="
