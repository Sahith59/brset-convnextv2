#!/bin/bash
#SBATCH --job-name=br_s3_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/smoke_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/frozen/launch_gate.py
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/frozen/train_controls.py \
  --arm C3_equal_domain_target_label --seed 0 --workers 4 --smoke \
  --output research/codex/step3/smoke
