#!/bin/bash
# Fair-baseline sweep for causal VMD (attic/RESULTS-superseded.md section 7, item 1).
# The NVMD>VMD claim currently rests on ONE untuned VMD config (alpha=2000,K=12).
# This sweeps alpha x K at the matched window W=96 so the baseline gets the same
# tuning courtesy NVMD got in section 6.
#
# Source CSVs are the same ones v3static modes were generated from, so row counts
# stay aligned (17099 / 16529) and the paired comparison remains valid.
set -u
cd /Users/lindseyf/sa_market

SRC18=../sa_market2/VMD_modes_with_residual_2018_2018.csv
SRC19=../sa_market2/VMD_modes_with_residual_2019_2019.csv

for K in 8 12 16; do
  for A in 250 500 1000 2000 4000 8000; do
    tag="vmdsw_a${A}_k${K}"
    if [ -f "${tag}_2019_2019.csv" ]; then echo "SKIP $tag (exists)"; continue; fi
    for yr in 2018 2019; do
      src=$SRC18; [ "$yr" = 2019 ] && src=$SRC19
      python3 -u generate_modes.py causal-vmd --csv "$src" \
        --output "${tag}_${yr}_${yr}.csv" \
        --K "$K" --alpha "$A" --window 96 --n-jobs 7 \
        >> vmd_sweep.log 2>&1 || echo "FAILED $tag $yr" >> vmd_sweep.log
    done
    echo "=== done $tag ===" >> vmd_sweep.log
  done
done
echo "=== VMD SWEEP GENERATION COMPLETE ===" >> vmd_sweep.log
