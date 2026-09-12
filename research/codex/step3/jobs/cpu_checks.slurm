#!/bin/bash
#SBATCH --job-name=br_s3_checks
#SBATCH --account=trends517s113
#SBATCH --partition=qTRD
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/checks_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/checks_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2/research/codex/step3/frozen
/home/users/sthummala2/.conda/envs/retfound/bin/python -m unittest -v test_sampler.py
/home/users/sthummala2/.conda/envs/retfound/bin/python preflight.py
