#!/bin/bash
#SBATCH --job-name=br_advisor_images
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/reports/private_build_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/reports/private_build_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/build_private_advisor_visuals.py
