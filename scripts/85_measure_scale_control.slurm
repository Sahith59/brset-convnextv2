#!/bin/bash
#SBATCH --job-name=chrscale
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/chrscale_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/chrscale_%j.err
#
# CPU only. Scale control for the target re-measurement.
#
# Three resolutions now exist for the same 400+400 images: native, a plain
# squash to 512, and the trainer's Resize((560,560)) plus 512 crop. The first
# two differ only in sampling pitch. The last two differ only by the crop, which
# removes the outer annulus where the retina meets the black surround. That rim
# is a hard step and script 74 is auditing how much of the measured Laplacian
# variance it contributes, so separating the two effects is necessary before
# anyone describes the sharpness gap as a camera property.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/76_measure_chromatic.py --n 400 --size 512 \
      --out results/chromatic_stats_sq512_nocrop.json
