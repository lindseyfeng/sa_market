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
# home is 20 GB and half of it is already gone; a torch env is ~3 GB.
if [ ! -L "$HOME/.conda" ]; then
  mkdir -p "$HOME/scratch/.conda"
  cp -a "$HOME/.conda/." "$HOME/scratch/.conda/" 2>/dev/null || true
  rm -rf "$HOME/.conda" && ln -s "$HOME/scratch/.conda" "$HOME/.conda"
fi
mkdir -p "$HOME/scratch/.cache/pip"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"

module load anaconda3/2023.03
conda create -y -n "$ENV_NAME" python=3.12
source activate "$ENV_NAME"
# Plain PyPI. The Linux x86_64 wheel there is already CUDA-enabled, and
# `--index-url https://download.pytorch.org/whl/cu124` is worse than
# unnecessary: it *replaces* PyPI, so torch's own build dependencies cannot be
# resolved and the install dies on `No matching distribution found for
# flit_core`. Use --extra-index-url if a specific CUDA build is ever needed.
pip install torch numpy pandas scipy
python -c "import torch; assert torch.version.cuda, 'got a CPU-only torch'"

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
