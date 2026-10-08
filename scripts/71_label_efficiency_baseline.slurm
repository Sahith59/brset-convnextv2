#!/bin/bash
#SBATCH --job-name=lebase
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/lebase_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/lebase_%j.err
#
# LABEL-EFFICIENCY BASELINE CURVE. The denominator for the whole synthesis study.
#
# Why this exists. Experiment 0 (all 3,402 real target images pooled with BRSET)
# already matches the best model trained directly on target data, 0.8455 against
# 0.8456 at matched best-cutoff. That setting is saturated, which is why five
# successive architectures all landed inside the 0.0199 seed-noise floor. There
# is no headroom left to compete for.
#
# The headroom is on the DATA axis, not the architecture axis. Real handheld
# images are expensive to collect and label; BRSET images are already paid for.
# So the question worth asking is not "can a better architecture beat pooling"
# but "how many REAL target labels does pooling actually need, and can synthetic
# target images replace them".
#
# This script measures the first half of that: performance as a function of the
# number of real target images, with NO synthesis. Every synthesis arm is scored
# against this curve at the same budget.
#
# Protocol note carried into the paper. Validation stays the full 725-image
# target val set at every budget, so the curve isolates training-set size. The
# honest headline therefore quotes TOTAL label cost (train budget + 725), and a
# budgeted-validation ablation is listed as follow-up work.
#
# Usage:
#   sbatch --export=ALL,BUDGET=500,SEED=0 scripts/71_label_efficiency_baseline.slurm
#
# Reference points already measured:
#   budget 3402, seeds 0/1/2 -> DR F1 0.8250 / 0.8449 / 0.8396 (tuned threshold)
#   seed-noise floor          -> 0.0199 DR F1 and 9 patients

set -euo pipefail
: "${BUDGET:?set BUDGET}"
: "${SEED:=0}"
: "${BSEED:=0}"

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
TASK="lebase_b${BUDGET}_s${SEED}"
echo "=== ${TASK}: budget=${BUDGET} seed=${SEED} budget_seed=${BSEED} ==="

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget "${BUDGET}" --budget_seed "${BSEED}" \
  --task "${TASK}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
