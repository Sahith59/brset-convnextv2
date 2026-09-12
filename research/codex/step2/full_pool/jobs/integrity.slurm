#!/bin/bash
#SBATCH --job-name=br_integrity
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/integrity_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/integrity_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step2/full_pool/frozen/check_image_duplicates.py
