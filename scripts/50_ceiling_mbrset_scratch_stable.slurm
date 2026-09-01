#!/bin/bash
#SBATCH --job-name=mb_stable
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/mb_stable_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/mb_stable_%j.err
#
# mBRSET from scratch with a stable schedule.
#
# The first attempt collapsed: its learning rate reached peak 5e-5 exactly at
# epoch 3 and validation AUC fell from 0.961 to 0.487, below chance. Warmup is
# 3 epochs, which is 4,263 optimizer steps on BRSET but only 636 on the much
# smaller mBRSET. Fixed here with lr 1.5e-5 and warmup 8.
# Result: DR F1 0.8239, no collapse.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py --model convnextv2_large.fcmae_ft_in22k_in1k_384 --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 --output_dir /home/users/sthummala2/brset-convnextv2/results --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel --epochs 40 --warmup_epochs 8 --lr 1.5e-5 --task mbrset_scratch_stable
