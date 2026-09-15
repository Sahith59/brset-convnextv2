#!/bin/bash
#SBATCH --job-name=br_s5_error_gate
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/appearance_error_gate_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/appearance_error_gate_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/appearance_error_gate.py
