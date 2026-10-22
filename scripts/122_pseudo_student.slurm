#!/bin/bash
#SBATCH --job-name=pstu
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/pstu_%x_%j.out
#
# SELF-TRAINING ON UNLABELLED mBRSET (+ optional synthetic twins). Copy of 114 with --pseudo_manifest.
# JOINT TRAINING WITH SYNTHETIC mBRSET (Dr. Ye, 8 Oct: "image synthesis converting
# BRSET to synthesized mBRSET and also have the joint training").
#
# Byte-for-byte the label-efficiency baseline (scripts/71_label_efficiency_baseline.slurm,
# runs lebase_b{BUDGET}_s{SEED}) PLUS --synth_dir: every BRSET training image is
# replaced by its generator-rendered mBRSET twin with p=0.5, redrawn on each load.
# Same eyes, same labels, same step count, same 250-image real subset (budget_seed 0)
# -- the only difference from lebase is the camera rendering of the BRSET images.
#
# Two questions, both against existing 3-seed baselines:
#   BUDGET=0    BRSET+synthetic  vs  lebase_b0   (0.9007): does synthesis help with NO target labels?
#   BUDGET=250  BRSET+synthetic+250 real  vs  lebase_b250 (0.9219) and lebase_b500 (0.9360):
#               can synthetic images stand in for real target labels?
# Generator choice (gen_both vs gen_tgt) is made on mBRSET VALIDATION (val_predictions.npz).
set -euo pipefail
: "${PSEUDO:?}"; : "${BUDGET:=0}"; : "${SEED:?}"; : "${BSEED:=0}"; : "${SYNTH_P:=0.25}"; : "${SD:=}"; : "${TASK:?}"
SYN=(); [ -n "$SD" ] && SYN=(--synth_dir "$SD" --synth_p "$SYNTH_P")
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
echo "=== ${TASK}: pseudo=${PSEUDO} synth=${SD:-none} p=${SYNTH_P} budget=${BUDGET} seed=${SEED} ==="
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
  ${SYN[@]+"${SYN[@]}"} --pseudo_manifest "$PSEUDO" \
  --task "${TASK}" \
  --output_dir /data/users3/sthummala2/brset_gen/cls_runs
