#!/bin/bash
#SBATCH --job-name=joint
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/joint_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/joint_%j.err
#
# Aggregated BRSET + mBRSET training.
#
# Per Dr. Ye's directive that training may use anything so long as the test
# split stays untouched. Train is 14,774 images (BRSET 11,372 + mBRSET 3,402);
# validation and test are mBRSET only, so the target domain is never
# contaminated.
# Result: DR F1 0.8250 with 27 of 159 missed, the fewest of any model, though
# statistically tied with fine-tuning (paired bootstrap, p = 0.47).

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py --model convnextv2_large.fcmae_ft_in22k_in1k_384 --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 --output_dir /home/users/sthummala2/brset-convnextv2/results --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint --epochs 25 --warmup_epochs 3 --lr 3e-5 --task brset_mbrset_joint
