#!/bin/bash
#SBATCH --job-name=ep50
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/ep50_%j.out
#
# DOES A LONGER SCHEDULE HELP? Dr. Ye asked directly (3 Sep): "did you try 50 or
# even 100 epoch?"
#
# This is NOT "25 more epochs on the end" -- that would be useless, because the
# LR is cosine-annealed to 2.0e-7 by epoch 24, a 150x drop from the 2.94e-5 peak.
# The model has already stopped updating. Passing --epochs 50 instead sets
# CosineAnnealingLR T_max = 47, which STRETCHES the schedule so the model spends
# far longer at a useful learning rate. A genuinely different trajectory.
#
# Prediction: no material gain. Across 13 runs the median improvement over the
# final 5 epochs is +0.0000, 10 of 13 peaked before epoch 22, and the peaks
# cluster at epochs 5-14 while the LR was still high -- the signature of a
# DATA-limited model, not an optimisation-limited one.
#
# Worth one run anyway: the advisor asked, a reviewer will ask, and "measured"
# beats "we reasoned it converges".
#
# Identical to lebase_b0 in every other respect, --no_oversample included, so it
# is directly comparable to the 0.9007 three-seed baseline. floor 0.0142.
set -euo pipefail
: "${SEED:=0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 50 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 \
  --task "ep50_b0_s${SEED}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
