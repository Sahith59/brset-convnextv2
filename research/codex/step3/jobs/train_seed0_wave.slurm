#!/bin/bash
#SBATCH --job-name=br_s3_wave
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/wave_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/wave_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1
srun --kill-on-bad-exit=1 \
  --output="/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/control_%t.out" \
  --error="/home/users/sthummala2/brset-convnextv2/research/codex/step3/logs/control_%t.err" \
  bash -lc '
    arms=(C1_equal_domain C2_target_label C3_equal_domain_target_label)
    arm="${arms[$SLURM_PROCID]}"
    /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/frozen/launch_gate.py
    /home/users/sthummala2/.conda/envs/retfound/bin/python research/codex/step3/frozen/train_controls.py \
      --arm "$arm" --seed 0 --workers 4 \
      --output "research/codex/step3/runs/${arm}_seed0"
  '
