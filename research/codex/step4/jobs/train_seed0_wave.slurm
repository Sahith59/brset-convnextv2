#!/bin/bash
#SBATCH --job-name=br_s4_seed0
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/seed0_wave_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/seed0_wave_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
srun --kill-on-bad-exit=1 \
  --output="/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/seed0_%j_%t.out" \
  --error="/home/users/sthummala2/brset-convnextv2/research/codex/step4/logs/seed0_%j_%t.err" \
  bash -lc '
    arms=(A1 A2 A3)
    arm="${arms[$SLURM_PROCID]}"
    /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step4/train_augmentation.py \
      --arm "$arm" --seed 0 --workers 6 --output "research/codex/step4/runs/${arm}_seed0"
  '
