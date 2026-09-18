#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# ON arm only: OFF already converged inside 10 epochs (early stop @6 and @7).
# 30-epoch cosine holds LR high far longer than the 10-epoch schedule, which
# collapsed to 6e-5 by epoch 7. Patience kept at 4 to match the OFF arms.
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 \
    --epochs 30 --batch 128 --w-sparse 0.001 --patience 4 --seed $s \
    --outdir ./runs_cmp_on30_s$s > cmp_on30_s$s.log 2>&1 &
done
wait
echo "=== COMPOUND30 COMPLETE ==="
grep -H "BEST test MAE" cmp_on30_s*.log
echo ""; sed -n '/Learned coupling/,$p' cmp_on30_s1.log
