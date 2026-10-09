#!/usr/bin/env bash
# One-time setup on PACE Phoenix. Run from a login node, after the GT VPN is up.
#
#   ssh <gtusername>@login-phoenix.pace.gatech.edu
#   git clone https://github.com/lindseyfeng/sa_market.git ~/scratch/sa_market
#   cd ~/scratch/sa_market && ./scripts/pace_setup.sh
#
# Everything lives under scratch. home is 15 GB and conda will fill it.
set -euo pipefail

ENV_NAME="${ENV_NAME:-samkt}"
ln -sfn "$HOME/scratch/.conda" "$HOME/.conda" 2>/dev/null || {
  mkdir -p "$HOME/scratch/.conda"; ln -sfn "$HOME/scratch/.conda" "$HOME/.conda"; }

module load anaconda3
conda create -y -n "$ENV_NAME" python=3.12
source activate "$ENV_NAME"
# The CUDA wheel, not the default CPU one. cu124 matches the drivers on the
# A100/H100 nodes at the time of writing; check `nvidia-smi` on a GPU node if
# torch reports no CUDA.
pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
pip install --quiet numpy pandas scipy

python -c "import torch; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
echo
echo "The panel is gitignored because it is rebuildable, but its source is not"
echo "in the repo. Copy it over from the laptop, then build the joint panel:"
echo
echo "  # on the laptop:"
echo "  scp ~/sa_market/data/raw/compound_unfiltered_2018_2022.csv \\"
echo "      <gtusername>@login-phoenix.pace.gatech.edu:~/scratch/sa_market/data/raw/"
echo
echo "  # back here:"
echo "  python -m data.make_joint_panel"
