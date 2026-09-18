#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# CONCAT arm: own modes (lossless) + purely-exogenous mix -> 2K head inputs.
# Plus a MATCHED OFF control at the same budget, because the earlier OFF runs
# used patience 4 on a 10-epoch cosine and stopped at epoch 6-7; comparing a
# 30-epoch/patience-10 concat arm against those would not be a fair contest.
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 --concat 1 \
    --epochs 30 --batch 128 --w-sparse 0.001 --patience 10 --seed $s \
    --outdir ./runs_cat_s$s > cat_s$s.log 2>&1 &
done
wait
echo "=== CONCAT ARM DONE ==="
grep -H "BEST test MAE" cat_s*.log
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 0 \
    --epochs 30 --batch 128 --patience 10 --seed $s \
    --outdir ./runs_off30_s$s > off30_s$s.log 2>&1 &
done
wait
echo "=== CONCAT COMPLETE ==="
grep -H "BEST test MAE" cat_s*.log off30_s*.log
echo ""; sed -n '/Learned coupling/,$p' cat_s1.log
