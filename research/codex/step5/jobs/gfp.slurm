#!/bin/bash
#SBATCH --job-name=br_s5_gfp
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/gfp_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/gfp_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=8
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/run_gfp.py
