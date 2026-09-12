#!/bin/bash
#SBATCH --job-name=br_step3_counts
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step3/design_counts_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step3/design_counts_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/compute_design_counts.py
