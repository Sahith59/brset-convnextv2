#!/bin/bash
#SBATCH --job-name=br_gallery_audit
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/reports/gallery_audit_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/reports/gallery_audit_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/audit_gallery_sources.py
