#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/diag_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/diag_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/90_diagnose_translator.py --n 60
