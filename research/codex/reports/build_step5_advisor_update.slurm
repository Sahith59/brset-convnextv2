#!/bin/bash
#SBATCH --job-name=br_s5_advisor
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/reports/advisor_s5_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/reports/advisor_s5_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/build_research_record.py
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/build_step5_advisor_update.py
