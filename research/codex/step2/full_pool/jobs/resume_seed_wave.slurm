#!/bin/bash
#SBATCH --job-name=br_resume_wave
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/resume_wave_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/resume_wave_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
seed="$1"
export TRAINING_SEED="$seed"
srun --kill-on-bad-exit=1 \
  --output="/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/resume_seed${seed}_%t.out" \
  --error="/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/resume_seed${seed}_%t.err" \
  bash -lc '
    arms=(B0 B1 B2)
    arm="${arms[$SLURM_PROCID]}"
    /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step2/full_pool/frozen/launch_gate.py
    /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step2/full_pool/frozen/train_baseline.py train \
      --arm "$arm" --seed "$TRAINING_SEED" --workers 4 --resume \
      --output "research/codex/step2/full_pool/runs/${arm}_seed${TRAINING_SEED}"
  '
