#!/bin/bash
#SBATCH --job-name=mbrset_ml_convnextv2
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/train_mbrset_ml_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/train_mbrset_ml_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2

# Same winning config as the BRSET run: ConvNeXt V2 Large @ 512, weighted
# oversampling + focal loss (gamma=2.0, no alpha) - only the dataset changes.
python scripts/07_train_convnextv2_multilabel.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel \
  --nb_classes 2 \
  --input_size 512 \
  --resize_size 560 \
  --batch_size 16 \
  --accum_iter 4 \
  --epochs 40 \
  --warmup_epochs 3 \
  --lr 5e-5 \
  --ckpt_freq 5 \
  --focal_gamma 2.0 \
  --task convnextv2_large_mBRSET_multilabel_512 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
