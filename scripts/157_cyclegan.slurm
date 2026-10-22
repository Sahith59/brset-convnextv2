#!/bin/bash
#SBATCH --job-name=cgan
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUH
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/cgan_%x_%j.out
#
# CycleGAN baselines (research/codex D107). MODE = train | translate_ab | translate_ba;
# VARIANT = plain | residual; EXTRA = any further flags (e.g. --identity, --run_name, --iters).
# train resumes from brset_gen/cyclegan/<run>/latest.pt after a requeue.
set -euo pipefail
: "${MODE:?}"; : "${VARIANT:=plain}"; : "${EXTRA:=}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
echo "=== cyclegan ${MODE} ${VARIANT} ${EXTRA} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
"$PY" scripts/157_cyclegan.py --mode "$MODE" --variant "$VARIANT" $EXTRA
