#!/bin/bash
#SBATCH --job-name=br_summary3
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/summary3_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/summary3_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/frozen
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/home/users/sthummala2/.conda/envs/retfound/bin/python summarize_all_seeds.py
