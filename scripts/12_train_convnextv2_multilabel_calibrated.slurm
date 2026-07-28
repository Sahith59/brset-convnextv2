#!/bin/bash
#SBATCH --job-name=brset_ml_calib
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/train_ml_calib_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/train_ml_calib_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2

python scripts/07_train_convnextv2_multilabel.py \
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
  --ckpt_freq 5 \
  --focal_gamma 1.0 \
  --focal_alpha_weighted \
  --task convnextv2_large_BRSET_multilabel_512_calibrated \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
