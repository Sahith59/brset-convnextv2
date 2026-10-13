#!/bin/bash
#SBATCH --job-name=gencache
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gencache_%j.out
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/108_build_gen_cache.py --workers 24
