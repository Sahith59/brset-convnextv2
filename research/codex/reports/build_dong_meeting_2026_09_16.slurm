#!/bin/bash
#SBATCH --job-name=br_dong_meeting
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/reports/private/dong_meeting_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/reports/private/dong_meeting_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
# This environment supplies python-pptx and Pillow only; the deck builder loads no model.
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/build_dong_meeting_2026_09_16.py
