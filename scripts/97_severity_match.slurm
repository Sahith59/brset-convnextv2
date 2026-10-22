#!/bin/bash
#SBATCH --job-name=sevm
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/sevm_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/sevm_%j.err
#
# SEVERITY MATCHING. The experiment the diagnosis implies.
#
# The binary DR label collapses an ordinal scale and the two cohorts sit at
# opposite ends of it. Among positives, verified 10 Oct on the PhysioNet
# metadata: BRSET is 14.6 percent mild and 36.6 percent proliferative, mBRSET
# is 24.0 and 18.7.
#
# Model performance is a pure function of grade. Zero-shot AUC by ICDR grade:
#   grade 1  0.7735      grade 3  0.9973
#   grade 2  0.9390      grade 4  0.9963
# and of the 42 missed patients, 24 are grade 1 and 18 are grade 2, with ZERO
# missed above grade 2. The model is near perfect on advanced disease in both
# domains; the entire transfer gap sits in early disease.
#
# This is orthogonal to appearance: all nine appearance statistics correlate
# with grade at |rho| <= 0.11. That is why matching 13 appearance statistics to
# |z| <= 0.32 could not help and why forcing it hurt. It also explains why the
# earlier class-balance control closed only 3 percent: marginal prevalence
# already matches, it is the CONDITIONAL severity distribution that differs.
#
# This run reweights SOURCE positives to the target severity mix. No target
# image, no target label, no architecture change. Total positive weight is
# preserved so prevalence is held fixed and only composition moves.
#
# Baselines on the same test set:
#   zero-shot, 3-seed mean   DR AUC 0.9007 (per seed 0.8906 / 0.9009 / 0.9104)
#   pooled 3402 real, mean   DR AUC 0.9463
#   gap                      0.0456
#   noise floors             DR AUC 0.0142, DR F1 0.0199
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
  --target_budget 0 --severity_match \
  --task "sevmatch_s${SEED}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
