#!/bin/bash
#SBATCH --job-name=dsn
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/dsn_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/dsn_%j.err
#
# EXPERIMENT 1, per Dr. Ye's routing proposal.
#
# Domain Separation Networks (Bousmalis et al., NeurIPS 2016) on the aggregated
# BRSET + mBRSET training set. Shared encoder plus a private encoder per domain,
# with an orthogonality constraint between them and a gradient-reversed domain
# classifier on the shared branch.
#
# Everything except the architecture matches the plain-joint run this must beat,
# so the comparison is clean:
#   plain joint    DR F1 0.8250, 27/159 missed | ME F1 0.8649, 15/63 missed
#   best finetune  DR F1 0.8355, 32/159 missed
#
# beta 0.075 and gamma 0.25 are the weights used in the DSN paper.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/46_domain_separation.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint \
  --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 \
  --proj_dim 512 --beta_diff 0.075 --gamma_sim 0.25 \
  --focal_gamma 2.0 --label_smoothing 0.1 --ema_decay 0.999 \
  --amp_dtype bf16 --tta flip4 --thr_bootstrap 200 \
  --num_workers 8 --seed 0 \
  --task dsn_joint \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
