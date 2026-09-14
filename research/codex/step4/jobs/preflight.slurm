#!/bin/bash
#SBATCH --job-name=br_s4_preflight
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/preflight_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/preflight_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step4/preflight.py
