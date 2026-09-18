#!/bin/bash
# Waits for run_vmd_sweep.sh, then ridge-screens the whole alpha x K grid
# against NVMD v3static.  Winners get promoted to benchmark_seeds.py by hand.
set -u
cd /Users/lindseyf/sa_market

while pgrep -f "generate_modes.py causal-vmd" >/dev/null; do sleep 60; done
echo "=== generation finished, screening ===" 

MARGS=()
for K in 8 12 16; do
  for A in 250 500 1000 2000 4000 8000; do
    tag="vmdsw_a${A}_k${K}"
    if [ -f "${tag}_2018_2018.csv" ] && [ -f "${tag}_2019_2019.csv" ]; then
      MARGS+=(--method "VMD a${A} k${K}" "${tag}_2018_2018.csv" "${tag}_2019_2019.csv")
    else
      echo "MISSING $tag"
    fi
  done
done
MARGS+=(--method "NVMD v3static" v3static_modes_2018_2018.csv v3static_modes_2019_2019.csv)

python3 -u ridge_screen.py "${MARGS[@]}" --ref "NVMD v3static" --window 96 --horizon 1
echo "=== SCREEN COMPLETE ==="
