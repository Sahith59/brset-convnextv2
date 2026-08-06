#!/bin/bash
#SBATCH --job-name=brset_focal
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/strongfocal_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/strongfocal_%j.err

# Companion to 31_train_strong_baseline.slurm. Identical in every respect
# except the loss: plain symmetric focal loss (gamma=2.0), the loss Dr. Ye
# named and the one behind the existing 0.869 F1. The single-model
# improvements (EMA, 4-way TTA, bootstrap-smoothed thresholds) are applied
# here too, so the two runs differ ONLY in the loss function and whichever
# wins on validation becomes the reported baseline.

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2

python scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 \
  --input_size 512 \
  --resize_size 560 \
  --batch_size 16 \
  --accum_iter 4 \
  --epochs 40 \
  --warmup_epochs 3 \
  --lr 5e-5 \
  --weight_decay 0.1 \
  --drop_path 0.3 \
  --mixup_alpha 0.2 \
  --label_smoothing 0.1 \
  --loss focal \
  --focal_gamma 2.0 \
  --ema_decay 0.999 \
  --tta flip4 \
  --thr_bootstrap 200 \
  --ckpt_freq 10 \
  --task convnextv2_large_BRSET_strong_baseline_focal \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
