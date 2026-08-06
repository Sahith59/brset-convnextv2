#!/bin/bash
#SBATCH --job-name=brset_strong
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/strong_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/strong_%j.err

# Single strong BRSET baseline: one ConvNeXt V2 Large, two labels
# (diabetic_retinopathy + macular_edema). Imbalance is handled by Asymmetric
# Loss + weighted oversampling; EMA/TTA/bootstrap thresholds tighten the
# operating point without adding a second model.

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
  --loss asl \
  --asl_gamma_neg 4.0 \
  --asl_gamma_pos 0.0 \
  --asl_clip 0.05 \
  --ema_decay 0.999 \
  --tta flip4 \
  --thr_bootstrap 200 \
  --ckpt_freq 10 \
  --task convnextv2_large_BRSET_strong_baseline \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
