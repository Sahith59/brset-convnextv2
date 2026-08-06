#!/bin/bash
#SBATCH --job-name=cnv2_smoke
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/smoke_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/smoke_%j.err
source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound
cd /home/users/sthummala2/brset-convnextv2
python scripts/30_train_strong_baseline.py \
  --model convnextv2_tiny.fcmae_ft_in22k_in1k --input_size 224 --resize_size 256 \
  --batch_size 32 --accum_iter 1 --epochs 1 --warmup_epochs 0 \
  --thr_bootstrap 50 --ckpt_freq 0 \
  --task SMOKE_TEST --output_dir /home/users/sthummala2/brset-convnextv2/results
