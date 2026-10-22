#!/bin/bash
#SBATCH --job-name=gates
#SBATCH --account=trends517s113
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gates_%x_%j.out
# D091 gates on synthetic folders DIRS: copy check (132) then label fidelity (134), tag TAG.
set -euo pipefail
: "${DIRS:?}"; : "${TAG:?}"
cd /home/users/sthummala2/brset-convnextv2
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
"$PY" -u scripts/132_copy_check.py --tag "$TAG" --synth_dir $DIRS
"$PY" -u scripts/134_label_gate.py --tag "$TAG" --synth_dir $DIRS
