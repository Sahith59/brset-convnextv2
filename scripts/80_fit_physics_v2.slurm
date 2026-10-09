#!/bin/bash
#SBATCH --job-name=fitv2
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/fitv2_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/fitv2_%j.err
#
# CPU only. Does not touch the eight GPU training jobs or the GPU authorisation.
# Depends on the chromatic measurement, which supplies the fit targets.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/79_fit_physics_v2.py --targets results/chromatic_stats_native.json \
      --out results/physics_v2_fit.json
