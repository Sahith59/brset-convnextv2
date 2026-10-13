#!/bin/bash
#SBATCH --job-name=synth
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/synth_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/synth_%j.err
#
# THE DECISIVE DOWNSTREAM EXPERIMENT.
#
# Identical to the label-efficiency baseline at the same budget, with exactly
# one difference: every training image is passed through the trained two-stage
# translator before normalisation. Same backbone, same augmentation, same
# epochs, same learning rate, same seed, same validation and test sets. Val and
# test are never translated; they are real target images.
#
# Baseline to beat, lebase_b0_s0 (zero real target training labels):
#   DR AUC 0.9104, val-threshold DR F1 0.7774, 42 of 159 missed
# Pooling with all 3,402 real target images reaches AUC 0.9463 / F1 0.8365.
# Seed noise floor: 0.0199 DR F1. AUC is the headline metric because it needs
# no threshold.
#
# Reading the result: synthesis has to move AUC materially toward 0.9463 to
# matter. Anything inside the noise floor is not a result.
set -euo pipefail
: "${BUDGET:=0}"
: "${SEED:=0}"
: "${CKPT:=/home/users/sthummala2/brset-convnextv2/results/xlate_v2_noartifact/translator_last.pth}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
TASK="synth_b${BUDGET}_s${SEED}_${TMODE:-prob}${TP:-1.0}"
echo "=== ${TASK}: budget=${BUDGET} seed=${SEED} translator=${CKPT} ==="

"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget "${BUDGET}" --budget_seed 0 \
  --translator_ckpt "${CKPT}" --translate_p "${TP:-1.0}" --translate_mode "${TMODE:-prob}" \
  --task "${TASK}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
