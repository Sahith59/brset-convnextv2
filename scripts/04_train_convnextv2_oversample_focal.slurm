#!/bin/bash
#SBATCH --job-name=brset_convnextv2_of
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/train_of_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/train_of_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2

python scripts/02_train_convnextv2.py \
  --model convnextv2_base.fcmae_ft_in22k_in1k \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_icdr5 \
  --nb_classes 5 \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --ckpt_freq 10 \
  --use_weighted_sampler \
  --focal_gamma 2.0 \
  --task convnextv2_base_BRSET_icdr5_oversample_focal_finetune \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
