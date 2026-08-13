#!/bin/bash
#SBATCH --job-name=s1_asl
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/s1_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/s1_%A_%a.err

# Stage 1: sweep the asymmetric-loss hyperparameters, as Dr. Ye asked.
#
# Stage 0 established two things that shape this design:
#   (a) removing the weighted oversampler IMPROVES results (focal macro-F1
#       0.8175 -> 0.8305), so the sweep runs without it;
#   (b) ASL at its published defaults loses to focal in BOTH sampler
#       conditions, so the over-correction hypothesis alone does not explain
#       the gap -- the defaults themselves may simply be wrong for K=2.
#
# Range justification (Ridnik et al., ASL, ICCV 2021):
#   gamma_neg: paper uses 4 for MS-COCO (3.76% positive) and raises it to 7
#     for Open Images because imbalance is higher. Our DR label is 6.6%
#     positive -- LESS imbalanced than MS-COCO -- so by the paper's own
#     scaling logic the optimum should sit BELOW 4. We sweep 1,2,3,5.
#   gamma_pos: paper's default is 0, but Appendix H reports gamma_pos=1 was
#     optimal for detection, and CheXFusion (ICCV-W 2023, CXR-LT winner) uses
#     gamma_pos=1 on medical multi-label data. Tested at gamma_neg=4.
#   clip: paper's Fig. 7 shows the optimal margin depends on gamma, and its
#     second role is discarding suspected mislabels. BRSET is
#     ophthalmologist-graded and clean, so the margin may cost more than it
#     buys. Tested at 0.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

COMMON="--model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_multilabel \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 20 --warmup_epochs 3 \
  --lr 5e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed 0 --no_oversample --loss asl \
  --output_dir /home/users/sthummala2/brset-convnextv2/results"

case $SLURM_ARRAY_TASK_ID in
  0) P="--asl_gamma_neg 1.0 --asl_gamma_pos 0.0 --asl_clip 0.05"; T="s1_gn1" ;;
  1) P="--asl_gamma_neg 2.0 --asl_gamma_pos 0.0 --asl_clip 0.05"; T="s1_gn2" ;;
  2) P="--asl_gamma_neg 3.0 --asl_gamma_pos 0.0 --asl_clip 0.05"; T="s1_gn3" ;;
  3) P="--asl_gamma_neg 5.0 --asl_gamma_pos 0.0 --asl_clip 0.05"; T="s1_gn5" ;;
  4) P="--asl_gamma_neg 4.0 --asl_gamma_pos 1.0 --asl_clip 0.05"; T="s1_gp1" ;;
  5) P="--asl_gamma_neg 2.0 --asl_gamma_pos 0.0 --asl_clip 0.0";  T="s1_gn2_clip0" ;;
esac

echo "=== Stage 1 run $SLURM_ARRAY_TASK_ID : $T ($P) ==="
"$PY" scripts/30_train_strong_baseline.py $COMMON $P --task "$T"
