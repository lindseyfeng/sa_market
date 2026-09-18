#!/bin/bash
# Wait for v3 training, then: generate modes -> leakage gate -> multi-seed benchmark.
set -u
cd /Users/lindseyf/sa_market

while pgrep -f train_nvmd_v3 >/dev/null; do sleep 20; done
echo "=== TRAINING FINISHED ==="

for f in nvmd_v3_static.log nvmd_v3_fastband.log; do
  echo "--- $f ---"
  grep -E "Saved best|Early stopping|No best state|Traceback|Error" "$f" | tail -3
  grep -A9 "Final band table" "$f" | tail -9
done

echo ""
echo "=== GENERATING V3 MODES ==="
# NB: no associative arrays -- macOS ships bash 3.2, which lacks `declare -A`.
OK=()
for pair in "v3static:runs_nvmd_v3_static" "v3fast:runs_nvmd_v3_fastband"; do
  tag="${pair%%:*}"
  ck="${pair##*:}/best.pt"
  if [ ! -f "$ck" ]; then echo "SKIP $tag: no $ck"; continue; fi
  good=1
  for yr in 2018_2018 2019_2019; do
    python3 generate_modes.py nvmd --model "$ck" \
      --csv "../sa_market2/VMD_modes_with_residual_${yr}.csv" \
      --output "${tag}_modes_${yr}.csv" --edge-pad 0 \
      || { echo "FAILED $tag $yr"; good=0; }
  done
  [ $good -eq 1 ] && OK+=("$tag")
done
echo "usable: ${OK[*]:-none}"
[ ${#OK[@]} -eq 0 ] && { echo "ABORT: no v3 modes generated"; exit 1; }

MARGS=()
for tag in "${OK[@]}"; do
  MARGS+=(--method "NVMD ${tag}" "${tag}_modes_2018_2018.csv" "${tag}_modes_2019_2019.csv")
done

echo ""
echo "=== AR LEAKAGE GATE ==="
python3 ar_probe.py \
  --method "Causal VMD" causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
  --method "NVMD v2" nvmd_modes_2018_2018_pad0.csv nvmd_modes_2019_2019_pad0.csv \
  "${MARGS[@]}"

echo ""
echo "=== MULTI-SEED BENCHMARK (5 seeds, Linear/MLP/LSTM, 10 epochs) ==="
# 10 epochs: every model in this project has been flat by ~epoch 10 (v2 hit
# 14.65 at ep3 and 14.02 at ep19; v3-static plateaued by ep10).  patience=4
# drops a seed as soon as it stops improving rather than burning the budget.
python3 benchmark_seeds.py \
  --method "Causal VMD" causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
  --method "NVMD v2" nvmd_modes_2018_2018_pad0.csv nvmd_modes_2019_2019_pad0.csv \
  "${MARGS[@]}" \
  --ref "Causal VMD" --seeds 5 --window 48 --epochs 10 --patience 4 --skip-cnn-bilstm

echo ""
echo "=== PIPELINE COMPLETE ==="
