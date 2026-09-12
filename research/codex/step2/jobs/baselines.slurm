#!/bin/bash
#SBATCH --job-name=br_baseline
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/logs/baseline_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/logs/baseline_%A_%a.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
arm="B${SLURM_ARRAY_TASK_ID}"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step2/frozen/train_baseline.py train \
  --arm "$arm" --seed 0 --workers 4 --output "research/codex/step2/runs/${arm}_seed0"
