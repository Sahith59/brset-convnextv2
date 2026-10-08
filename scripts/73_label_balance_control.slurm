#!/bin/bash
#SBATCH --job-name=balctl
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/balctl_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/balctl_%j.err
#
# THE DECISIVE CONTROL FOR THE WHOLE SYNTHESIS PAPER.
#
# Dr. Ye raised this on 3 September: "It's not just about the image quality, but
# also the label distribution. You have more balanced data in Setup B." It was
# never tested, and it is the single assumption the ISBI claim rests on.
#
# The problem. BRSET train is 6.6 percent DR-positive, mBRSET is 23.4 percent.
# Pooling therefore does TWO things at once: it adds handheld-camera appearance,
# AND it more than doubles the absolute positive count, 748 to 1545. Those are
# confounded in every result measured so far.
#
# Why it decides the paper. Synthetic handheld images are made from BRSET images
# and INHERIT BRSET'S LABELS at 6.6 percent. They add appearance but not a single
# new positive. So:
#   - If the pooling gain is mostly device appearance, synthesis can reproduce it
#     and the label-efficiency claim is sound.
#   - If the pooling gain is mostly class balance, synthesis CANNOT reproduce it
#     and the headline claim fails no matter how good the translator is.
#
# The test. Train on BRSET only, zero mBRSET images, but resample so DR positives
# are drawn at mBRSET's 23.4 percent rate. Everything else matches the budget-0
# baseline exactly.
#
# Reference points on the same 732-image test set:
#   budget 0, no rebalancing   DR F1 0.7842, 50 of 159 missed   (running, 4572237)
#   pooling, all 3402 real     DR F1 0.8455 at best cutoff
#   seed noise floor           0.0199 DR F1, 9 patients
#
# Reading the result: if this run lands near 0.84, appearance is NOT the story
# and the plan must change before the translator is built. If it stays near 0.79,
# device adaptation is the story and the synthesis direction is sound.

set -euo pipefail
: "${SEED:=0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
TASK="balctl_s${SEED}"

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 --target_pos_rate 0.234 \
  --task "${TASK}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
