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
#
# Targets are the TRAINER-PITCH ones (presize 560, crop 512), not the native
# ones used until 9 Oct. Laplacian variance is scale dependent, so fitting
# 512-pixel tensors against native-resolution targets understated the sharpness
# gap by about 3x and would have made the model's own noise self-audit pass
# spuriously. Produce the targets with scripts/85_measure_sq512.slurm.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
TGT=results/chromatic_stats_sq512.json
if [ ! -f "$TGT" ]; then
  echo "missing $TGT: run scripts/85_measure_sq512.slurm first" >&2
  exit 1
fi
"$PY" scripts/79_fit_physics_v2.py --targets "$TGT" \
      --presize 560 --size 512 \
      --out results/physics_v2_fit.json
