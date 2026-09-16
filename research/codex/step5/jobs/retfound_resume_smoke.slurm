#!/bin/bash
#SBATCH --job-name=br_s5_rf_resume
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_resume_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/retfound_resume_%j.err
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
cd /home/users/sthummala2/brset-convnextv2
BASE=/data/users3/sthummala2/brset-codex/step5
rm -rf "$BASE/retfound_smoke_full" "$BASE/retfound_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_retfound_matched.py --smoke --seed 0 --workers 4 --output "$BASE/retfound_smoke_full"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_retfound_matched.py --smoke --seed 0 --workers 4 --stop-after 3 --output "$BASE/retfound_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_retfound_matched.py --smoke --seed 0 --workers 4 --resume --output "$BASE/retfound_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/verify_retfound_resume.py
