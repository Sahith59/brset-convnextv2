#!/bin/bash
#SBATCH --job-name=br_s4_refit
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/refit_degradation_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/refit_degradation_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step4/refit_degradation.py
