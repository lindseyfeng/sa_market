#!/bin/bash
# Joint multi-region forecasting on one GPU. Submit from the repo root:
#
#   sbatch --account=gts-<yours> scripts/pace_joint.sbatch
#
# Find the charge account with `pace-quota`. inferno is Phoenix's primary QoS
# and allows 3 days of GPU wallclock, so the whole arm sweep fits in one job.
#
# Why this is worth moving off the laptop: on 3 CPU threads one epoch of the
# 5-target LSTM arm took 450-580 s, and under swap pressure it reached 23,560 s
# -- the run had to be killed. The model is small (650 k parameters, 96 steps,
# batch 64), so a single V100 should put an epoch in the tens of seconds. That
# is the difference between one 15-epoch single-seed arm per evening and the
# three seeds, longer schedules and loss sweeps the conclusions actually need.
#SBATCH --job-name=sa-joint
#SBATCH --qos=inferno
# One core and one GPU, with the type left open. Measured on this cluster, what
# decides the wait is the size of the hole, not the QoS or the partition:
#
#   no GPU                    submit -> start   3 s
#   8 cores + 1 GPU                            54 s
#   1 core  + 1 GPU                         3 m 51 s   (started)
#   embers (priority 0)                     still pending after 5 min
#   pinned gpu-v100 / gpu-rtx6000           still pending after 5 min
#
# Asking for four cores had Slurm hand over eight with 64 GB and a billing
# weight of 10,261. The CPU only feeds the GPU here, so that was a large hole
# waited for to no purpose. `interactive-gpu` is not submittable with sbatch.
#SBATCH --partition=gpu-a100,gpu-l40s,gpu-h100,gpu-rtx6000
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pace_%j.out
#SBATCH --mail-type=NONE

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results preds

# Call the environment's interpreter directly. `source activate <env>` does not
# reliably switch interpreters in a non-interactive shell: it silently left the
# base python 3.10 in place, so pip installed torch into ~/.local and put 5.2 GB
# into a 20 GB home directory instead of the 15 TB scratch the env lives on.
PY="$HOME/scratch/.conda/envs/${ENV_NAME:-samkt}/bin/python"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import torch; print('cuda available:', torch.cuda.is_available())"

# --threads matches the cores Slurm gave us; a V100 reserves 12.
"$PY" -u -m experiments.run_joint \
  --panel data/raw/compound_joint_2018_2022.csv \
  --train-year 2018,2019,2020 --test-year 2021 \
  --horizon "${HORIZON:-6}" \
  --arms "${ARMS:-single:SA1_price,joint,joint_nocouple}" \
  --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 \
  --loss "${LOSS:-l1}" \
  --seeds "${SEEDS:-1,2,3}" --epochs "${EPOCHS:-30}" --patience 30 \
  --threads 1 --workers 0 \
  --save-preds "preds/pace_h${HORIZON:-6}" \
  --out "results/pace_h${HORIZON:-6}.json"
