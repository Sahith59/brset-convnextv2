#!/bin/bash
#SBATCH --job-name=dsn_g005
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/dsn_g005_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/dsn_g005_%j.err
#
# Experiment 1b: domain-adversarial weight reduced from 0.25 to 0.05. Tests whether the adversarial term is scrubbing device information the decision still needs, which would explain better ranking alongside worse operating points.
#
# Baseline to beat, from the plain aggregated run:
#   DR F1 0.8250, 27 of 159 missed | ME F1 0.8649, 15 of 63 missed
# Experiment 1 as first run reached DR F1 0.8053 with 37 missed, so it lost
# ten patients while gaining AUC.

#
# RESULT (task dsn_joint_gamma005): DR F1 0.8227, AUC 0.9404, 36 of 159 missed | ME F1 0.8889, 11 of 63 missed.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/46_domain_separation.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --proj_dim 512 \
  --focal_gamma 2.0 --label_smoothing 0.1 --ema_decay 0.999 \
  --amp_dtype bf16 --tta flip4 --thr_bootstrap 200 \
  --num_workers 8 --seed 0 --beta_diff 0.075 --gamma_sim 0.05 --select_metric f1 \
  --task dsn_joint_gamma005 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
