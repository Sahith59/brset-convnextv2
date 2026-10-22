#!/bin/bash
#SBATCH --job-name=lebase_rt
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUH
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/lebase_rt_%j.out
# D107a: the checkpoints of lebase_b0_s1/s2 were deleted (home-quota cleanup, 10 Oct); only their predictions remain.
# Retrain them with the EXACT command of scripts/71_label_efficiency_baseline.slurm (BUDGET=0), but into
# cls_runs/lebase_b0_s{SEED}_rt so the stored results/lebase_b0_s* are never touched. Used only by the D107
# test-time arm, where each model is compared with itself on untranslated vs translated phone photos.
set -euo pipefail
: "${SEED:?}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
TASK="lebase_b0_s${SEED}_rt"
echo "=== ${TASK} on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --target_budget 0 --budget_seed 0 \
  --task "${TASK}" \
  --output_dir /data/users3/sthummala2/brset_gen/cls_runs
