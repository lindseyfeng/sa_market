#!/bin/bash
set -u
cd /Users/lindseyf/sa_market
# h=6 (3 h ahead).  h=1 was saturated: persistence 14.40 vs causal VMD+LSTM 14.43.
# At h=6 the naive floor is 32.61 and a linear ridge already reaches 24.32 (-25.4%),
# with the exogenous panel contributing -0.96 -- so both headroom and exogenous
# signal are here.  XFILTER vs matched OFF control, 2 seeds, same 30ep/patience-10.
for s in 1 2; do
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 1 --xfilter 1 \
    --horizon 6 --epochs 30 --batch 128 --w-sparse 0.001 --patience 10 --seed $s \
    --outdir ./runs_h6xf_s$s > h6xf_s$s.log 2>&1 &
  python3 -u train_nvmd_st.py --panel compound_2018_2022.csv --coupling 0 \
    --horizon 6 --epochs 30 --batch 128 --patience 10 --seed $s \
    --outdir ./runs_h6off_s$s > h6off_s$s.log 2>&1 &
done
wait
echo "=== H6 COMPLETE ==="
grep -H "BEST test MAE" h6xf_s*.log h6off_s*.log
