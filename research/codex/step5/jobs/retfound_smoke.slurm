#!/bin/bash
#SBATCH --job-name=br_s5_rf_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_smoke_%j.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/retfound_smoke.py "$@"
