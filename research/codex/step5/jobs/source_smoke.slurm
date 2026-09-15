#!/bin/bash
#SBATCH --job-name=br_s5_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/source_smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step5/logs/source_smoke_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
base=/data/users3/sthummala2/brset-codex/step5
rm -rf "$base/source_smoke_full" "$base/source_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_source_only.py train --seed 0 --workers 4 --smoke --output "$base/source_smoke_full"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_source_only.py train --seed 0 --workers 4 --smoke --stop-after 3 --output "$base/source_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/train_source_only.py train --seed 0 --workers 4 --smoke --resume --output "$base/source_smoke_resume"
/home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step5/verify_source_smoke.py

