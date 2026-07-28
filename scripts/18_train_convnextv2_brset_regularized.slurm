#!/bin/bash
#SBATCH --job-name=brset_ml_reg
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/train_brset_reg_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/train_brset_reg_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2

# Same winning loss config (gamma=2.0, no alpha, oversampling), now with
# regularization aimed directly at the confirmed train-test overfitting gap:
# drop_path (previously unset), higher weight decay, multi-label mixup,
# label smoothing.
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
  --weight_decay 0.1 \
  --drop_path 0.3 \
  --mixup_alpha 0.2 \
  --label_smoothing 0.1 \
  --ckpt_freq 5 \
  --focal_gamma 2.0 \
  --task convnextv2_large_BRSET_multilabel_512_regularized \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
