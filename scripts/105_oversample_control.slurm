#!/bin/bash
#SBATCH --job-name=osctl
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/osctl_%j.out
#
# THE MISSING CONTROL for severity matching.
#
# --severity_match reweights the WeightedRandomSampler. --no_oversample discards
# that sampler, so the two flags together are a silent no-op: sevmatch_*_ns came
# back BITWISE identical to lebase_b0 (0.9104020547268596 to 16 digits).
#
# Severity matching therefore only exists under oversampling, where it scored
# 0.9094 -- but there has never been a baseline WITH oversampling to compare it
# to. This run is that baseline: identical to lebase_b0 in every way except that
# --no_oversample is omitted.
#
#   this run (oversampling, no severity match)   vs   sevmatch (0.9094)
#
# That difference is the severity-matching effect, measured for the first time.
# Everything else about these two is held fixed. floor 0.0142.
set -euo pipefail
: "${SEED:=0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 \
  --task "osctl_b0_s${SEED}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
