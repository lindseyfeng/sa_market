#!/usr/bin/env bash
# Submit one job per arm, all at once.
#
# Measured on this cluster the wait is set by the size of the hole, not the QoS:
# one core with one GPU started in 3 m 51 s where eight cores waited longer and
# embers (priority 0) never landed. Six one-core jobs therefore backfill into
# six separate holes in parallel, which finishes the sweep in about the time of
# its slowest arm rather than the sum of all six.
#
# Each job writes its own result file. A shared --out would have six processes
# racing on one JSON, and the resume logic reads it at startup.
#
# One seed. Three would triple the cost for a margin this sweep is not yet
# trying to resolve; seeds come after an arm is worth repeating.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results preds

ACCT="${ACCT:-gts-sd111}"
H="${HORIZON:-6}"
EPOCHS="${EPOCHS:-15}"

#   single           the single-task baseline
#   joint            bands + per-band coupling, LSTM head
#   joint_nocouple   the coupling ablation; A_k frozen at identity
#   panel            THE no-decomposition control -- same head, raw window. The
#                    only arm that isolates the decomposition, because a
#                    partition-of-unity bank is an invertible linear map and a
#                    ridge cannot tell bands from the raw window at all.
#   +film            the FiLM gate: own bands multiplied by a function of the
#                    exogenous state, the one part a linear reader cannot copy
#   +film+ctx336     the same with the bank seeing 336 steps, so the 168 h weekly
#                    cycle exists in its output at all
ARMS=${ARMS:-"single:SA1_price joint joint_nocouple panel/lstm joint/lstm+film joint/lstm+film+ctx336"}

for arm in $ARMS; do
  tag=$(echo "$arm" | tr '/:+' '___')
  sbatch --account="$ACCT" --qos=inferno \
    --job-name="h${H}_${tag}" \
    --partition=gpu-a100,gpu-v100,gpu-h100,gpu-l40s,gpu-rtx6000 \
    --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --mem-per-cpu=32G \
    --time=02:00:00 \
    --output="logs/h${H}_${tag}_%j.out" \
    --wrap "cd \$SLURM_SUBMIT_DIR && \
      nvidia-smi --query-gpu=name --format=csv,noheader && \
      \$HOME/scratch/.conda/envs/samkt/bin/python -u -m experiments.run_joint \
        --panel data/raw/compound_joint_2018_2022.csv \
        --train-year 2018,2019,2020 --test-year 2021 --horizon $H \
        --arms '$arm' --seeds 1 --epochs $EPOCHS --patience $EPOCHS \
        --w-dev 1.0 --w-bias 0.1 --w-sparse 1e-4 --loss l1 \
        --threads 1 --workers 0 \
        --save-preds preds/h${H} --out results/h${H}_${tag}.json" \
    | sed "s/^/  $arm -> /"
done
echo
squeue -u "$USER" -o "%.10i %.26j %.3t %.6M %.16R"
