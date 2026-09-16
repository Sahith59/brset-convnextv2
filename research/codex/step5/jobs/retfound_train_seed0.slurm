#!/bin/bash
#SBATCH --job-name=br_s5_rf_train0
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_train0_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_train0_%j.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export PYTHONWARNINGS=ignore::UserWarning
cd /home/users/sthummala2/brset-convnextv2
OUT=/data/users3/sthummala2/brset-codex/step5/retfound_matched_seed0
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_retfound_matched.py --seed 0 --workers 4 --output "$OUT"
