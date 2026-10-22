#!/bin/bash
#SBATCH --job-name=udatest
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/udatest_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/udatest_%j.err
#
# CPU unit tests for the unsupervised-domain-adaptation trainer
# (scripts/94_uda_align.py). Synthetic tensors and synthetic JPEGs only: no
# credentialed images, no GPU. Runs here rather than on the login node because
# it decodes images and builds models (AGENTS.md).

set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/94_test_uda_align.py
