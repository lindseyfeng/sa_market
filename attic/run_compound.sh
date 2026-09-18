#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 \
    --epochs 10 --batch 128 --w-sparse 0.001 --seed $s \
    --outdir ./runs_cmp_on_s$s > cmp_on_s$s.log 2>&1 &
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 0 \
    --epochs 10 --batch 128 --seed $s \
    --outdir ./runs_cmp_off_s$s > cmp_off_s$s.log 2>&1 &
  wait
  echo "seed $s done: ON=$(grep -oE 'BEST test MAE = [0-9.]+' cmp_on_s$s.log | tail -1) OFF=$(grep -oE 'BEST test MAE = [0-9.]+' cmp_off_s$s.log | tail -1)"
done
echo "=== COMPOUND COMPLETE ==="
grep -H "BEST test MAE" cmp_*_s*.log
echo ""; sed -n '/Learned coupling/,$p' cmp_on_s1.log
