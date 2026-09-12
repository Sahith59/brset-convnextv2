#!/bin/bash
#SBATCH --job-name=br_assess_seed
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/assessment_seed_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/assessment_seed_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
seed="$1"
/home/users/sthummala2/.conda/envs/retfound/bin/python \
  research/codex/step2/full_pool/frozen/assess_replication.py \
  --seed "$seed" \
  --batch-gate "research/codex/step2/full_pool/batch_gate_seed${seed}.json" \
  --output "research/codex/step2/full_pool/baseline_assessment_seed${seed}.json"
