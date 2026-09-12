#!/bin/bash
#SBATCH --job-name=br_full_checks
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/preflight_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/preflight_%j.err
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
cd /home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/frozen
py=/home/users/sthummala2/.conda/envs/retfound/bin/python
"$py" prepare_manifest.py
"$py" test_baseline.py
"$py" check_image_duplicates.py
