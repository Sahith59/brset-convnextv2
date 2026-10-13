#!/bin/bash
#SBATCH --job-name=extmild
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/extmild_ns_%j.out
#
# IMPORT REAL MILD DISEASE -- the properly powered test of the severity thesis.
#
# The measured bottleneck: BRSET train holds ~115 ICDR grade-1 images. Every
# intervention that only reweights or re-supervises them (severity matching,
# ordinal head) landed inside noise and cost macular oedema 4-7% AUC, because
# reweighting cannot create diversity that is not in the data.
#
# The 250-label oracle was UNDERPOWERED, not refuted: only 59 images (0.51% of
# training data) differed between its mild and severe arms. It did show the
# predicted order (mild 0.9172 > severe 0.9124) but both sat below a random mix
# (0.9219), and the whole spread was inside the 0.0142 floor.
#
# DDR supplies 5,107 real mild cases on the same ICDR scale. Each arm adds 1,000
# external POSITIVES, matched count, so severity is the only thing that varies
# -- a 17x larger perturbation than the oracle applied.
#
#   ARM=mild    grade 1-2 only   (the hypothesis)
#   ARM=severe  grade 3-4 only   (the control: same new camera, wrong severity)
#   ARM=mixed   proportional     (because random beat both extremes at n=250)
#
# DDR carries no oedema label, so that label is MASKED for external rows rather
# than written as 0 -- verified: gradient exactly 0 on the masked term.
#
# No target labels used, so this stays comparable to zero-shot 0.9007.
# floor 0.0142   |   250 random target labels = 0.9219
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
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 \
  --extra_manifest results/extra_manifest.csv \
  --extra_n 1000 --extra_severity "${ARM}" --extra_seed 0 \
  --task "extmild_${ARM}_s${SEED}_ns" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
