#!/bin/bash
#SBATCH --job-name=ordaux
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/ordaux_ns_%j.out
#
# ORDINAL AUXILIARY HEAD -- fixes the REPRESENTATION, not the distribution.
#
# The binary DR label collapses ICDR 0-4, so a grade-1 eye and a grade-4 eye
# carry an identical target. BRSET's positives are 43.8% grade 3-4; mBRSET's
# are 25.9%. The model therefore builds one "positive" prototype out of mostly
# florid disease and applies it to a cohort whose positives are 74.1% grade 1-2.
#
# Measured: zero-shot AUC 0.7735 at grade 1 vs 0.9963 at grade 4; all 42 missed
# patients are grade 1 or 2, none above.
#
# Four extra outputs give P(grade >= k), k=1..4, keeping a monotone severity
# axis in the representation. Source grades only -- no target label is used,
# so this stays a zero-shot comparison against the 0.9007 three-seed baseline.
#
#   ARM=ord   ordinal head alone        (representation fix)
#   ARM=both  ordinal + severity_match  (representation + distribution)
#
# baseline  zero-shot DR AUC 0.9007 (3-seed mean)   noise floor AUC 0.0142
set -euo pipefail
: "${SEED:=0}"
: "${ARM:=ord}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

EXTRA=""
if [ "$ARM" = "both" ]; then EXTRA="--severity_match"; fi

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 \
  --ordinal_aux --ordinal_weight 0.3 ${EXTRA} \
  --task "ordaux_${ARM}_s${SEED}_ns" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
