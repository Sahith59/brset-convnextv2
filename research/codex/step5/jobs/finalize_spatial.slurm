#!/bin/bash
#SBATCH --job-name=br_s5_head_final
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/spatial_final_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/spatial_final_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/finalize_spatial_diagnostic.py
