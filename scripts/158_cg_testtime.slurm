#!/bin/bash
#SBATCH --job-name=cgtt
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/cgtt_%x_%j.out
# D107 test-time arm: BRSET-only classifiers on CycleGAN-translated (or untranslated: TAG=ident) phone val/test.
set -euo pipefail
: "${TAG:?}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
echo "=== cgtt ${TAG} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
"$PY" scripts/158_cg_testtime.py --tag "$TAG"
