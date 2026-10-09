#!/bin/bash
# Deliberately tiny: this job exists to measure two unknowns before any real
# work is charged to gts-sd111, which has 285 credits and no job history to
# infer a rate from.
#
#   1. the charge. `pace-quota` before and after gives the actual rate for one
#      V100 with 4 cores, which is the only way to size the real sweep.
#   2. the epoch time. Every estimate so far was extrapolated from 3 CPU threads
#      on a laptop, where an epoch ran 450-580 s and once reached 23,560 s.
#
# 4 cores, not the 12 the ICE guidance suggests: that ratio is an ICE rule and
# this model is one small LSTM, so paying for cores it cannot use would only
# distort the rate measurement. --time is a hard 15-minute cap.
#SBATCH --job-name=sa-smoke
#SBATCH --qos=inferno
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu:V100:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --mail-type=NONE

set -x
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results preds
module load anaconda3/2023.03
source activate samkt

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"

# 2,000 train windows and 3 epochs: enough to time an epoch and prove the GPU is
# actually being used, far too few for any number to mean anything.
python -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 --horizon 6 \
  --arms "single:SA1_price,joint" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 --loss l1 \
  --seeds 1 --epochs 3 --max-windows 2000 \
  --threads 4 --workers 2 \
  --out results/smoke_gpu.json --save-preds preds/smoke_gpu
echo SMOKE_DONE
