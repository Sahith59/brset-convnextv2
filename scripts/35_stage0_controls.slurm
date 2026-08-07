#!/bin/bash
#SBATCH --job-name=s0_ctrl
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/s0_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/s0_%A_%a.err

# Stage 0 of the asymmetric-loss investigation: four controls that decide
# whether ASL lost because it is the wrong loss, or because it was applied on
# top of weighted oversampling that had already rebalanced the data past parity.
#
# Measured: the sampler lifts the positive rate the loss sees from 6.6% to
# 61.0% (DR) and 2.4% to 38.5% (ME). ASL's defaults (gamma_neg=4, m=0.05) were
# tuned on MS-COCO at 3.76% positive. So with the sampler on we are applying a
# correction for 26.6:1 imbalance to data that is already positive-majority.
#
# Published basis for the hypothesis:
#   Zhang et al., Bag of Tricks, AAAI 2021 - re-sampling and re-weighting
#     together give "similar or even worse accuracy than using them alone"
#   Wu et al., Distribution-Balanced Loss, ECCV 2020, Sec 3.1 - re-balanced
#     weighting after re-sampling
#   Ridnik et al., ASL, ICCV 2021 - designed as a REPLACEMENT for resampling
#
# Runs 2 and 3 repeat the existing configs with a new seed, so we can tell
# whether the original 0.869 vs 0.834 gap exceeds run-to-run variance at all.

# Call the env's interpreter directly. `conda activate` silently failed inside
# the array job, leaving the system python (no numpy) on PATH.
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

COMMON="--model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 20 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results"

case $SLURM_ARRAY_TASK_ID in
  0) EXTRA="--loss focal --focal_gamma 2.0 --no_oversample --seed 0"
     TASK="s0_focal_nosamp" ;;
  1) EXTRA="--loss asl --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 --no_oversample --seed 0"
     TASK="s0_asl_nosamp" ;;
  2) EXTRA="--loss focal --focal_gamma 2.0 --seed 1"
     TASK="s0_focal_samp_seed1" ;;
  3) EXTRA="--loss asl --asl_gamma_neg 4.0 --asl_gamma_pos 0.0 --asl_clip 0.05 --seed 1"
     TASK="s0_asl_samp_seed1" ;;
esac

echo "=== Stage 0 run $SLURM_ARRAY_TASK_ID : $TASK ==="
"$PY" scripts/30_train_strong_baseline.py $COMMON $EXTRA --task "$TASK"
