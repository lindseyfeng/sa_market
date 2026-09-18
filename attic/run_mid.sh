#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# Fill the h=6 -> h=48 gap.  The ridge probe puts the exogenous collapse between
# h=12 and h=24 (gain -0.97 -> -0.10), so these two points locate the crossover
# and turn a 3-point inverted-U into an actual curve.
for h in 12 24; do
  for s in 1 2; do
    python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 --xfilter 1 \
      --horizon $h --epochs 30 --batch 128 --w-sparse 0.001 --patience 10 --seed $s \
      --outdir ./runs_h${h}xf_s$s > h${h}xf_s$s.log 2>&1 &
    python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 0 \
      --horizon $h --epochs 30 --batch 128 --patience 10 --seed $s \
      --outdir ./runs_h${h}off_s$s > h${h}off_s$s.log 2>&1 &
  done
  wait
  echo "=== h=$h NEURAL DONE ==="
  grep -H "BEST test MAE" h${h}xf_s*.log h${h}off_s*.log
done
for h in 12 24; do
  python3 -u benchmark_seeds.py \
    --method "Causal VMD"   causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
    --method "NVMD v3static" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv \
    --ref "Causal VMD" --seeds 3 --window 96 --horizon $h \
    --epochs 30 --patience 6 --skip-cnn-bilstm > vmd_h${h}.log 2>&1
  echo "=== vmd h=$h done ==="
done
echo "=== MID COMPLETE ==="
