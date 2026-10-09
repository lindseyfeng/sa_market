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
# distort the rate measurement.
#
# Every GPU partition is deep -- 153 to 245 jobs pending across
# v100/a100/h100/h200/l40s -- and this model (650 k parameters, 96 steps) cannot
# use an H200 any better than a V100, so the only thing that matters is which
# gap opens first. Dropping the GPU *type* from --gres was not enough: Slurm
# still placed the job in gpu-v100, because partitions are defined by node
# feature, not chosen by gres. The partitions have to be listed explicitly for
# the scheduler to consider all of their gaps.
#SBATCH --job-name=sa-smoke
#SBATCH --qos=inferno
#SBATCH --partition=gpu-v100,gpu-a100,gpu-h100,gpu-h200,gpu-l40s
#SBATCH --nodes=1 --ntasks-per-node=4 --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --mail-type=NONE

set -x
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results preds
# Call the environment's interpreter directly. `source activate <env>` does not
# reliably switch interpreters in a non-interactive shell: it silently left the
# base python 3.10 in place, so pip installed torch into ~/.local and put 5.2 GB
# into a 20 GB home directory instead of the 15 TB scratch the env lives on.
PY="$HOME/scratch/.conda/envs/${ENV_NAME:-samkt}/bin/python"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "SLURM gave: $SLURM_JOB_PARTITION on $SLURMD_NODENAME"
"$PY" -c "import torch; print('cuda', torch.cuda.is_available(), torch.version.cuda)"

# 2,000 train windows and 3 epochs: enough to time an epoch and prove the GPU is
# actually being used, far too few for any number to mean anything.
"$PY" -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 --horizon 6 \
  --arms "single:SA1_price,joint" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 --loss l1 \
  --seeds 1 --epochs 3 --max-windows 2000 \
  --threads 4 --workers 2 \
  --out results/smoke_gpu.json --save-preds preds/smoke_gpu
echo SMOKE_DONE
