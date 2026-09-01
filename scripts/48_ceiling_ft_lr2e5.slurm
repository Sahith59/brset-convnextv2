#!/bin/bash
#SBATCH --job-name=ft_lr2e5
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/ft_lr2e5_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/ft_lr2e5_%j.err
#
# Fine-tune learning-rate sweep, lr 2e-5.
#
# Part of establishing what mBRSET labels actually buy. The earlier 0.8317
# figure came from a single barely-tuned config at lr 1e-5 and was never a
# ceiling. Result: DR diseased-class F1 0.8355, best of the six target-trained
# models.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py --model convnextv2_large.fcmae_ft_in22k_in1k_384 --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 --output_dir /home/users/sthummala2/brset-convnextv2/results --warm_start_ckpt /home/users/sthummala2/brset-convnextv2/results/afl40_focal_control/checkpoint-best.pth --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel --epochs 25 --warmup_epochs 2 --lr 2e-5 --task mbrset_ft_lr2e5
