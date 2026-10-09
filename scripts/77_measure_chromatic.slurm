#!/bin/bash
#SBATCH --job-name=chroma
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/chroma_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/chroma_%j.err
#
# CPU-only. Uses no GPU, so it does not touch the eight training jobs or the
# three-concurrent-GPU authorisation.
#
# Measures the chromatic gap, which has never been measured, plus a
# fundus-disc-matched version of the sharpness comparison to remove the
# retinal-scale confound in the earlier claim.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
echo "### native resolution"
"$PY" scripts/76_measure_chromatic.py --n 400 --out results/chromatic_stats_native.json
echo; echo "### fundus disc matched to 1024 px (removes the retinal-scale confound)"
"$PY" scripts/76_measure_chromatic.py --n 400 --match_disc 1024 --out results/chromatic_stats_discmatched.json
