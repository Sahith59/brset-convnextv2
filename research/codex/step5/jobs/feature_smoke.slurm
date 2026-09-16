#!/bin/bash
#SBATCH --job-name=br_s5_feat_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/feature_smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/feature_smoke_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/extract_b1_features.py smoke --workers 4 --smoke-images 32
