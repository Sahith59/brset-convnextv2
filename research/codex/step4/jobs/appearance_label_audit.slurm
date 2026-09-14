#!/bin/bash
#SBATCH --job-name=br_s4_app_audit
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/appearance_label_audit_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/appearance_label_audit_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step4/appearance_label_audit.py
