#!/bin/bash
#SBATCH --job-name=p1_bf16
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/p1_bf16_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/p1_bf16_%j.err
#
# STEP 5. Rerun the selected Part-1 baseline in bf16.
#
# The original run (afl40_focal_control) diverged to NaN during epoch 26 under
# fp16 and produced non-finite weights from epoch 28 to 40. Its best checkpoint
# is from epoch 7, so the model everything currently rests on trained usefully
# for 7 of its 40 epochs.
#
# Cause: ConvNeXt V2's GRN takes an L2 norm over a 512x512 feature map, which
# can exceed the fp16 maximum of 65504. bf16 carries fp32's exponent range and
# cannot overflow there. A guard now also drops any non-finite batch instead of
# letting it poison the accumulated gradients.
#
# Everything else is unchanged from the selected configuration, so the result is
# directly comparable to the 40-epoch head-to-head table.
#   Numbers to beat: DR AUC 0.9906, DR class-macro F1 0.9374, 10/162 missed.
#
# batch 16 / accum 4 matches the afl40_g2_d60 run, which was stable on BRSET.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 40 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task part1_focal_bf16 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
