#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# Wait for the h=6 arms to clear before adding load (8 cores, 4 procs each).
while pgrep -f run_h6.sh >/dev/null; do sleep 30; done
echo "=== h6 clear, starting h=48 (24h ahead, DART decision point) ==="
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 --xfilter 1 \
    --horizon 48 --epochs 30 --batch 128 --w-sparse 0.001 --patience 10 --seed $s \
    --outdir ./runs_h48xf_s$s > h48xf_s$s.log 2>&1 &
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 0 \
    --horizon 48 --epochs 30 --batch 128 --patience 10 --seed $s \
    --outdir ./runs_h48off_s$s > h48off_s$s.log 2>&1 &
done
wait
echo "=== H48 NEURAL COMPLETE ==="
grep -H "BEST test MAE" h48xf_s*.log h48off_s*.log
# causal VMD at the same horizon, once the VMD h=1/h=6 sweep is clear
while pgrep -f run_vmd_h.sh >/dev/null; do sleep 30; done
python3 -u benchmark_seeds.py \
  --method "Causal VMD"   causal_vmd_2018_2018_w96.csv causal_vmd_2019_2019_w96.csv \
  --method "NVMD v3static" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv \
  --ref "Causal VMD" --seeds 3 --window 96 --horizon 48 \
  --epochs 30 --patience 6 --skip-cnn-bilstm > vmd_h48.log 2>&1
echo "=== H48 COMPLETE ==="
sed -n '/Val MAE, mean/,$p' vmd_h48.log
