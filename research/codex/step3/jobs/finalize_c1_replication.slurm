#!/bin/bash
#SBATCH --job-name=br_s3_c1_final
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/c1_final_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/c1_final_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/frozen/verify_c1_replication.py
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/reports/build_research_record.py
