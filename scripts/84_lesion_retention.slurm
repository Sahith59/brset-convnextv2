#!/bin/bash
#SBATCH --job-name=lesret
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/lesret_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/lesret_%j.err
#
# CPU ONLY. Touches no GPU, so it is outside the three-concurrent-GPU
# authorisation and cannot disturb the eight GPU jobs already running.
# Image decoding happens here on an allocated node, never on the login node.
#
# Three phases, in order, because a failure in the first makes the rest
# meaningless:
#   0  smoke: four images, controls only, so a bug costs one minute not one hour
#   1  TRAINER PITCH, presize 560 then a 512 crop. This is the resolution the
#      translator actually runs at in training (scripts 79, 81), so this is the
#      number that governs whether our training labels are being corrupted.
#   2  NATIVE resolution, same images. Lesion survival is resolution dependent
#      and native is the generous reference: if retention is poor even here the
#      problem is the operator, not the sampling grid.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
mkdir -p results logs
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # workers are per-image, not per-op

echo "### phase 0: smoke"
"$PY" scripts/84_lesion_retention.py --n_pos 2 --n_neg 2 --workers 4 \
  --only NULL_identity,NULL_jitter_0.002,POS_blur_sigma3,artifact_sigma0.015_amp0.40 \
  --out results/lesion_retention_smoke.json

echo; echo "### phase 1: trainer pitch (presize 560 -> crop 512)"
"$PY" scripts/84_lesion_retention.py --n_pos 80 --n_neg 80 --workers 8 --per_image \
  --presize 560 --size 512 --print_bins --out results/lesion_retention_sq512.json

echo; echo "### phase 2: native resolution"
"$PY" scripts/84_lesion_retention.py --n_pos 80 --n_neg 80 --workers 8 --per_image \
  --presize 0 --size 0 --print_bins --out results/lesion_retention_native.json
