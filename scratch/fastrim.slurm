#!/bin/bash
#SBATCH --job-name=fastrim
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/fastrim_%j.out
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scratch/fastrim.py 40
