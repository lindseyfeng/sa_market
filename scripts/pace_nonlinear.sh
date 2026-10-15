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
EPOCHS="${EPOCHS:-30}"
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
    --job-name="nl${H}_${tag}" \
    --partition=gpu-a100,gpu-l40s,gpu-h100,gpu-rtx6000 \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --mem-per-cpu=32G \
    --time="${WALL:-00:25:00}" \
    --output="logs/nl${H}_${tag}_%j.out" \
    --wrap "cd \$SLURM_SUBMIT_DIR && \
      nvidia-smi --query-gpu=name --format=csv,noheader && \
      $PY -u -m experiments.run_joint \
        --panel data/raw/compound_joint_2018_2022.csv \
        --train-year 2018,2019,2020 --test-year 2021 --horizon $H \
        --arms '$arm' --seeds 1 --epochs $EPOCHS --patience $EPOCHS \
        --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 --loss l1 $jac \
        --save-model runs/nl${H}/${tag} \
        --threads 1 --workers 0 \
        --save-preds preds/nl${H}/${tag} --out results/nl${H}_${tag}.json" \
    | while read -r line; do printf '  %-34s %s\n' "$spec" "$line"; done
done
echo
squeue -u "$USER" -o "%.10i %.30j %.3t %.6M %.16R"
