#!/bin/bash
#SBATCH --job-name=afl
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/afl_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/afl_%A_%a.err
#
# ASYMMETRIC FOCAL LOSS sweep, as directed by Dr. Ye.
#
#   L = -delta * y * log(p)  -  (1-delta) * (1-y) * p^gamma * log(1-p)
#
# The defining property: positives receive NO focal modulation. Only the
# abundant negatives are down-weighted. On a 15:1 (DR) and 40:1 (ME) skew this
# is the behaviour we want -- a rare positive's gradient is never discounted.
#
# Origin: Li, Kamnitsas & Glocker, MICCAI 2019 -> IEEE TMI (189 cites).
# delta term: Yeung et al., Comput Med Imaging Graph 2022 (686 cites), which
# recommends delta = 0.6.
#
# Grid rationale:
#   gamma in {1,2,3,4} - how hard to suppress easy negatives. Standard focal
#     uses 2; higher values suit heavier negative dominance, and ours is heavy.
#   delta in {0.6, 0.75} - 0.6 is the published recommendation; 0.75 tilts
#     further toward positives, which our 40:1 ME skew may want.
#
# Oversampling is OFF for every run. Stage 0 measured that the sampler lifts
# the effective positive rate from 6.6% to 61%, inverting the imbalance the
# loss is meant to handle, and that removing it improved focal macro-F1 from
# 0.8175 to 0.8305.
#
# Number to beat: focal loss, no oversampling, 20 epochs -> macro F1 0.8305
#   (DR F1 0.8711 / AUC 0.9919, ME F1 0.7899 / AUC 0.9955)

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

COMMON="--model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 20 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --loss afl \
  --output_dir /home/users/sthummala2/brset-convnextv2/results"

case $SLURM_ARRAY_TASK_ID in
  0) P="--afl_gamma 1.0 --afl_delta 0.60"; T="afl_g1_d60" ;;
  1) P="--afl_gamma 2.0 --afl_delta 0.60"; T="afl_g2_d60" ;;
  2) P="--afl_gamma 3.0 --afl_delta 0.60"; T="afl_g3_d60" ;;
  3) P="--afl_gamma 4.0 --afl_delta 0.60"; T="afl_g4_d60" ;;
  4) P="--afl_gamma 2.0 --afl_delta 0.75"; T="afl_g2_d75" ;;
  5) P="--afl_gamma 3.0 --afl_delta 0.75"; T="afl_g3_d75" ;;
esac

echo "=== Asymmetric Focal Loss sweep, run $SLURM_ARRAY_TASK_ID : $T ($P) ==="
"$PY" scripts/30_train_strong_baseline.py $COMMON $P --task "$T"
