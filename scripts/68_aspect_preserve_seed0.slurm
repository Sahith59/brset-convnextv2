#!/bin/bash
#SBATCH --job-name=ar_s0
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/ar_s0_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/ar_s0_%j.err
#
# ASPECT-RATIO FIX, seed 0 of 3.
#
# Measured 3 September 2026: every BRSET image is non-square (mean aspect 1.287,
# median native 2672x2056) and every mBRSET image is exactly square (1600x1600).
# The transform Resize((560,560)) forces both dimensions, so BRSET is compressed
# horizontally by about 23 percent while mBRSET passes through untouched. The
# model therefore trains on distorted retinas and is tested on undistorted ones.
# This is a geometric domain gap created in preprocessing, not by the cameras.
#
# --resize_mode preserve resizes the short side and centre crops, treating both
# datasets identically. Everything else matches Experiment 0 exactly.
#
# Three seeds, because the seed-variance control measured a noise floor of
# 0.0199 DR F1 and 9 missed patients. Single-run comparisons at this effect
# size cannot be interpreted.
#
# Baseline, Experiment 0 over seeds 0/1/2:
#   DR F1 0.8250 / 0.8449 / 0.8396   mean 0.8365
#   DR missed 27 / 31 / 36           mean 31.3

#
# RESULT (task aspect_preserve_seed0): DR F1 0.8283, AUC 0.9474, 36 of 159 missed | ME F1 0.8364, 17 of 63 missed.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 --resize_mode preserve \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task aspect_preserve_seed0 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
