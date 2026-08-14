#!/bin/bash
#SBATCH --job-name=afl40
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/afl40_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/afl40_%A_%a.err
#
# APPLES-TO-APPLES: Asymmetric Focal Loss vs plain focal loss, 40 epochs each.
#
# Every setting identical across the three runs except the loss function:
# same backbone, resolution, LR, schedule, regularization, EMA, TTA,
# threshold procedure, seed, and epoch count. Oversampling is OFF everywhere,
# because Stage 0 showed the sampler inverts the imbalance (effective positive
# rate 6.6% -> 61%) and that removing it improved focal.
#
# Run 2 is the control. Without it we could not tell whether any AFL gain came
# from the loss or from having dropped the oversampler.
#
# Numbers to beat -- the PPT baseline (focal WITH oversampling, 40 epochs):
#   DR AUC 0.9924, DR class-macro F1 0.9297
#
# Array is capped at %3 so at most three of the six L40S GPUs are used at once,
# leaving capacity for other users on the node.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

COMMON="--model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 40 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample \
  --output_dir /home/users/sthummala2/brset-convnextv2/results"

case $SLURM_ARRAY_TASK_ID in
  0) P="--loss afl --afl_gamma 2.0 --afl_delta 0.60"; T="afl40_g2_d60" ;;
  1) P="--loss afl --afl_gamma 3.0 --afl_delta 0.75"; T="afl40_g3_d75" ;;
  2) P="--loss focal --focal_gamma 2.0";             T="afl40_focal_control" ;;
esac

echo "=== 40-epoch head-to-head, run $SLURM_ARRAY_TASK_ID : $T ($P) ==="
"$PY" scripts/30_train_strong_baseline.py $COMMON $P --task "$T"
