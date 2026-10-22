#!/bin/bash
#SBATCH --job-name=zljudge
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/zljudge_%x_%j.out
# D099 clean BRSET-only judge: real BRSET train, checkpoint and cut-off chosen on REAL BRSET validation
# (no mBRSET label anywhere). Recipe identical to the lebase arms. Used only INSIDE generators (Translator-v2-clean).
set -euo pipefail
: "${SEED:=0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/zl_teacher_brset \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task "zl_teacher_brset_s${SEED}" \
  --output_dir /data/users3/sthummala2/brset_gen/cls_runs
