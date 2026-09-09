#!/bin/bash
#SBATCH --job-name=routed_condent
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/routed_condent_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/routed_condent_%j.err
#
# EXPERIMENT 2 (weight driven by measured conditional entropy of the label given each branch, with the gate trained to imitate it for inference)
#
# Built on the zero-adversarial-weight variant, not the published defaults,
# because the experiment 1 sweep showed those defaults hurt here:
#   weight 0.25 -> DR F1 0.8053, 37 missed
#   weight 0.00 -> DR F1 0.8232, 31 missed
#
# Baseline the gate must beat on its own: plain aggregated training,
#   DR F1 0.8250, 27 of 159 missed | ME F1 0.8649, 15 of 63 missed

#
# RESULT (task routed_condent): DR F1 0.8414, AUC 0.9480, 37 of 159 missed | ME F1 0.7692, 23 of 63 missed.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/52_entropy_routed.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --proj_dim 512 \
  --beta_diff 0.075 --gamma_sim 0.0 \
  --focal_gamma 2.0 --label_smoothing 0.1 --ema_decay 0.999 \
  --amp_dtype bf16 --tta flip4 --thr_bootstrap 200 \
  --num_workers 8 --seed 0 --select_metric f1 --route condent --lambda_aux 0.1 --lambda_mimic 1.0 --route_temp 1.0 \
  --task routed_condent \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
