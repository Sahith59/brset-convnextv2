#!/bin/bash
#SBATCH --job-name=mb_scratch
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/mb_scratch_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/mb_scratch_%j.err
#
# RUN 2 of 3. Train on mBRSET with the Part-1 recipe.
#
# The existing in-domain reference (F1 0.8146) came from the OLD script, which
# always enabled the weighted oversampler, had no EMA, used 2-way flip TTA and
# a plain argmax-F1 cutoff. Measured: that sampler drove mBRSET's DR positive
# rate from 23.4% to 66.5%, the same over-correction Part 1 identified and
# fixed on BRSET. So 0.8146 is an under-tuned reference, not a ceiling.
#
# This run applies the Part-1 recipe unchanged: oversampling OFF, EMA 0.999,
# 4-way flip TTA, bootstrap-smoothed cutoffs.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 40 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --task mbrset_part1recipe_scratch \
  --output_dir /home/users/sthummala2/brset-convnextv2/results

# bf16 rather than fp16: the Part-1 focal run diverged to NaN at epoch 26 under
# fp16 and trained uselessly for its last 12 epochs. bf16 has fp32's exponent
# range, so the GRN L2 norm at 512x512 cannot overflow.
