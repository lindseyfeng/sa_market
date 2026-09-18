#!/bin/bash
# NVMD v3 config sweep.  Two jobs at a time (8 cores, ~2 cores each leaves
# headroom).  10 epochs each -- everything in this project is flat by ~10.
set -u
cd /Users/lindseyf/sa_market

run () {  # tag, extra args...
  tag="$1"; shift
  if [ -f "runs_sweep_${tag}/best.pt" ]; then echo "skip $tag (done)"; return; fi
  python3 -u train_nvmd_v3.py --seq-len 96 --epochs 10 --patience 4 \
    --outdir "./runs_sweep_${tag}" "$@" > "sweep_${tag}.log" 2>&1
  echo "finished $tag: $(grep -E 'Saved best' "sweep_${tag}.log" | tail -1)"
}

# Baseline config is v3static: K=8 adapt=0 band-lr=3e-4 head=lstm.
run k12      --K 12 --adapt 0   --band-lr 1e-3 &
run k16      --K 16 --adapt 0   --band-lr 1e-3 &
wait
run adapt    --K 12 --adapt 0.5 --band-lr 1e-3 &
run linhead  --K 12 --adapt 0   --band-lr 1e-3 --head linear &
wait

echo "=== SWEEP TRAINING DONE ==="
grep -H "Saved best" sweep_*.log

echo ""
echo "=== GENERATING MODES ==="
OK=()
for tag in k12 k16 adapt linhead; do
  ck="runs_sweep_${tag}/best.pt"
  [ -f "$ck" ] || { echo "SKIP $tag"; continue; }
  good=1
  for yr in 2018_2018 2019_2019; do
    python3 generate_modes.py nvmd --model "$ck" \
      --csv "../sa_market2/VMD_modes_with_residual_${yr}.csv" \
      --output "sweep_${tag}_modes_${yr}.csv" --edge-pad 0 >/dev/null 2>&1 \
      || { echo "FAILED $tag $yr"; good=0; }
  done
  [ $good -eq 1 ] && OK+=("$tag")
done
echo "usable: ${OK[*]:-none}"
[ ${#OK[@]} -eq 0 ] && { echo "ABORT"; exit 1; }

MARGS=()
for tag in "${OK[@]}"; do
  MARGS+=(--method "v3-${tag}" "sweep_${tag}_modes_2018_2018.csv" "sweep_${tag}_modes_2019_2019.csv")
done

echo ""
echo "=== SWEEP BENCHMARK vs causal VMD ==="
python3 benchmark_seeds.py \
  --method "Causal VMD" causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
  --method "v3-static" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv \
  "${MARGS[@]}" \
  --ref "Causal VMD" --seeds 5 --window 48 --epochs 10 --patience 4 --skip-cnn-bilstm

echo "=== SWEEP COMPLETE ==="
