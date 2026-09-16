#!/bin/bash
#SBATCH --job-name=br_s5_heads
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/spatial_head_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/spatial_head_%A_%a.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_spatial_heads.py --head-seed "$SLURM_ARRAY_TASK_ID"
