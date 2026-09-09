#!/bin/bash
#SBATCH --job-name=seed1
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/seed1_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/seed1_%j.err
#
# THE MISSING CONTROL. Experiment 0 repeated with a different random seed and
# nothing else changed.
#
# Every architecture comparison in this project has produced differences of
# roughly 0.02 F1. Without knowing how much a plain re-run varies from seed
# alone, none of those differences can be interpreted. This measures the noise
# floor that every other result has to clear.
#
# Reference, seed 0: DR F1 0.8250, 27 of 159 missed | ME F1 0.8649, 15 of 63.

#
# RESULT (task joint_seed1): DR F1 0.8449, AUC 0.9519, 31 of 159 missed | ME F1 0.8224, 19 of 63 missed.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 1 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task joint_seed1 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
