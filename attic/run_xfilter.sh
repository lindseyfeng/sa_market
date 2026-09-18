#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# Cross-filter arm: own (lossless) + cross-band exo + FiLM-gated own -> 3K head inputs.
# Same protocol as the concat arm (30 ep, patience 10, 2 seeds) so the two are
# directly comparable; OFF30 and CONCAT already ran under it.
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 --xfilter 1 \
    --epochs 30 --batch 128 --w-sparse 0.001 --patience 10 --seed $s \
    --outdir ./runs_xf_s$s > xf_s$s.log 2>&1 &
done
wait
echo "=== XFILTER COMPLETE ==="
grep -H "BEST test MAE" xf_s*.log
echo ""; sed -n '/Learned coupling/,$p' xf_s1.log
