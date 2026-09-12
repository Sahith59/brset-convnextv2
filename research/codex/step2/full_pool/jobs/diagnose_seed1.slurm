#!/bin/bash
#SBATCH --job-name=br_diag_seed1
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --nodelist=arctrdagn039
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/diagnose_seed1_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/diagnose_seed1_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
/home/users/sthummala2/.conda/envs/retfound/bin/python \
  research/codex/step2/full_pool/frozen/diagnose_seed1_failure.py
