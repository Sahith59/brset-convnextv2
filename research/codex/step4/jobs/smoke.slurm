#!/bin/bash
#SBATCH --job-name=br_s4_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/smoke_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
for arm in A1 A2 A3; do
  /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step4/train_augmentation.py \
    --arm "$arm" --seed 0 --workers 4 --smoke --output "research/codex/step4/smoke/${arm}"
done
