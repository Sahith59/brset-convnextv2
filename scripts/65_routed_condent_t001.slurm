#!/bin/bash
#SBATCH --job-name=routed_condent_t001
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/routed_condent_t001_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/routed_condent_t001_%j.err
#
# EXPERIMENT 2, corrected temperature (0.01).
#
# The first condent run held alpha at 0.501 for all 25 epochs. A 44-second
# diagnostic (script 53) measured why: the conditional-entropy difference has
# std 0.0327 and spans -0.042 to +0.047, but dividing by temperature 1.0
# squashed alpha into [0.455, 0.554], so no routing decision was ever made.
# The temperature was wrong by roughly 30x. This was an implementation error,
# not evidence that the signal is uninformative.
#
# At temperature 0.03, 63.6 percent of samples land outside [0.4, 0.6].
# At 0.01, 86.9 percent do.
#
# Also of note: the mean difference is -0.0059, so the PRIVATE branch is very
# slightly more informative on average. The gate variant collapsed to
# shared-only, which this contradicts. Worth watching in these runs.
#
# Baseline to beat: plain aggregated training, DR F1 0.8250, 27 of 159 missed.

#
# RESULT (task routed_condent_t001): DR F1 0.8414, AUC 0.9468, 29 of 159 missed | ME F1 0.8621, 13 of 63 missed.

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
  --num_workers 8 --seed 0 --select_metric f1 \
  --route condent --lambda_aux 0.1 --lambda_mimic 1.0 --route_temp 0.01 \
  --task routed_condent_t001 \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
