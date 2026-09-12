#!/bin/bash
#SBATCH --job-name=br_checks
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/checks_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/checks_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step2/full_pool/frozen/test_baseline.py
