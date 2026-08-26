#!/bin/bash
#SBATCH --job-name=mb_ft
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/mb_ft_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/mb_ft_%j.err
#
# RUN 3 of 3. BRSET pre-train -> mBRSET fine-tune.
#
# mBRSET has only 3,402 training images against BRSET's 11,372. Training on
# mBRSET alone throws the larger dataset away. Warm-starting from the selected
# Part-1 BRSET model uses both, which on a small target set usually beats
# training from scratch. This is the strongest available estimate of what
# target-domain training can actually reach, and therefore the honest endpoint
# for the gap decomposition.
#
# Fine-tuning schedule: lr 1e-5 (one fifth of the from-scratch lr) over 25
# epochs, to adapt without erasing the pre-trained representation.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --warm_start_ckpt /home/users/sthummala2/brset-convnextv2/results/afl40_focal_control/checkpoint-best.pth \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 2 \
  --lr 1e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task mbrset_finetune_from_brset \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
