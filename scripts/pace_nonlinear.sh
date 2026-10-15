#!/usr/bin/env bash
# The nonlinear-decomposition sweep: does breaking the linear bank buy anything?
#
# Everything between the input and the LSTM used to be a matrix multiply, and
# the masks are a partition of unity, so sum_k x_k == x identically -- which is
# why ridge on bands equals ridge on the raw window to 0.0000 $/MWh. These arms
# are the switches that break that. All are exact no-ops at initialisation, so
# each can only depart from the linear bank if the objective pays for it.
#
#   joint/lstm                  the linear bank, in this batch, as the control
#   +gate                       magnitude-dependent spectral gain
#   +env                        analytic envelope + instantaneous frequency
#   +gate+env                   both
#   +gate+env+bgate             both, plus the band aggregator
#   +ad50                       input-dependent band edges (adapt=0.5)
#   +gate+env+bgate @ jac 0.25  the full stack on the best-measured alpha
#
# One job per arm so they backfill into separate holes; one seed, because a
# seed resolves 0.116 MAE here and this sweep is looking for more than that.
#
# Each job gets its own --out AND its own --save-preds. The second matters more
# than it looks: prediction filenames come from the arm name, and the last two
# entries are the *same arm* differing only in a CLI flag, so a shared preds
# directory would have them silently clobber each other's .npz.
#
# --save-model takes a directory, not a bare flag: passing it bare swallows the
# next argument. Checkpoints matter here. `gate_c` near zero means the arm
# trained back to the linear bank and its MAE says nothing about nonlinearity,
# which is exactly the reading the FiLM arm's 38.45-against-38.50 never got.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds

ACCT="${ACCT:-gts-sd111}"
H="${HORIZON:-6}"
# A tag for this submission. Without one, the runner's resume logic finds an
# existing result JSON and silently does nothing -- which is what happened to
# batch 2: joint/lstm and joint/lstm+env were resubmitted to get an in-batch
# control, skipped themselves, and left the batch comparing against a control
# trained on different hardware. Measured, that is worth 0.28 MAE here.
TAG="${RUNTAG:-nl}"
PANEL="${PANEL:-data/raw/compound_joint_2018_2022.csv}"
HZ="${HORIZONS:-}"
# --partition is NOT honoured on this cluster. Measured 2026-10-15: seven jobs
# submitted with --partition=gpu-a100,gpu-l40s,gpu-h100,gpu-rtx6000 and an
# untyped --gres=gpu:1; six of them ran in gpu-v100, which was not in the list,
# and died in seconds on
#
#   RuntimeError: cuDNN version 92400 is not compatible with devices with
#   SM < 7.5
#
# because a V100 is compute capability 7.0 and this torch ships kernels for
# CC >= 7.5 only. A node feature is a hard constraint and cannot be routed
# around: V100 nodes carry the feature `V100`, so requiring any of the others
# makes them unschedulable. The probe that carried this constraint landed on an
# H200. Keep --partition as a hint; --constraint is what actually binds.
EPOCHS="${EPOCHS:-30}"
FEATURES="${FEATURES:-A100|H100|H200|L40S|RTX6000}"
PY="\$HOME/scratch/.conda/envs/samkt/bin/python"

# "arm" or "arm@@alpha" for --w-jacobian.
SPECS=${SPECS:-"
joint/lstm
joint/lstm+gate
joint/lstm+env
joint/lstm+gate+env
joint/lstm+gate+env+bgate
joint/lstm+ad50
joint/lstm+gate+env+bgate@@0.25
"}

for spec in $SPECS; do
  arm="${spec%@@*}"
  jac=""
  [ "$spec" != "$arm" ] && jac="--w-jacobian ${spec##*@@}"
  # The tag has to separate the two same-arm entries, so it is built from the
  # whole spec, not the arm. `sed s|^|$arm |` would break here: arm names
  # contain '/'.
  tag=$(printf '%s' "$spec" | tr '/:+@.' '_____')
  sbatch --account="$ACCT" --qos=inferno \
    --job-name="${TAG}${H}_${tag}" \
    --partition=gpu-a100,gpu-l40s,gpu-h100,gpu-h200,gpu-rtx6000 \
    --constraint="$FEATURES" \
    --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 \
    --mem-per-cpu=32G \
    --time="${WALL:-00:30:00}" \
    --output="logs/${TAG}${H}_${tag}_%j.out" \
    --wrap "cd \$SLURM_SUBMIT_DIR && \
      nvidia-smi --query-gpu=name --format=csv,noheader && \
      $PY -u -m experiments.run_joint \
        --panel $PANEL \
        --train-year 2018,2019,2020 --test-year 2021 --horizon $H \
        ${HZ:+--horizons $HZ} --arms '$arm' --seeds 1 --epochs $EPOCHS --patience $EPOCHS \
        --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 --loss l1 $jac \
        --save-model runs/${TAG}${H}/${tag} \
        --threads 1 --workers 0 \
        --save-preds preds/${TAG}${H}/${tag} --out results/${TAG}${H}_${tag}.json" \
    | while read -r line; do printf '  %-34s %s\n' "$spec" "$line"; done
done
echo
squeue -u "$USER" -o "%.10i %.30j %.3t %.6M %.16R"
