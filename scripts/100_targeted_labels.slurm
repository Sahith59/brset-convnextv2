#!/bin/bash
#SBATCH --job-name=tgtlab
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/tgtlab_%j.out
#
# WHICH LABELS TO BUY, not how many.
#
# The gap is a severity-distribution gap, not an appearance gap: zero-shot AUC
# 0.7735 at ICDR 1 vs 0.9963 at ICDR 4, every missed patient at grade 1-2, and
# appearance orthogonal to grade at |rho| <= 0.11. Reweighting the SOURCE cannot
# fix it -- BRSET train holds only ~115 grade-1 positives, so upweighting shows
# the model the same images more often (measured: all three reweighting/ordinal
# arms came in BELOW baseline).
#
# So buy target labels instead -- but spend the budget where the model is blind.
# All three arms hold budget (250), positive count (59) and positive rate
# (23.6%) IDENTICAL. The only thing that varies is the severity of the positives
# bought:
#   ARM=mild     positives drawn from ICDR 1-2 only   (realised {1:17, 2:42})
#   ARM=severe   positives drawn from ICDR 3-4 only   (realised {3:22, 4:37})
#   random       already measured: lebase_b250 = 0.9219
#
# SEVERE IS THE CONTROL AND IT IS THE POINT. If mild > random > severe, severity
# is confirmed as the mechanism and "buy mild labels" follows. If mild ~= severe,
# the budget is all that matters and the severity story does not transfer to
# acquisition -- which we would then report as such.
#
# baseline  0 labels 0.9007   |   250 random 0.9219   |   noise floor 0.0142
set -euo pipefail
: "${SEED:=0}"
: "${ARM:=mild}"
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
  --target_budget 250 --budget_seed 0 --budget_severity "${ARM}" \
  --task "tgtlab_${ARM}_s${SEED}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
