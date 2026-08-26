#!/bin/bash
#SBATCH --job-name=xdev_eval
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/xdev_eval_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/xdev_eval_%j.err
#
# RUN 1 of 3. Re-run the BRSET -> mBRSET transfer test with the NEW Part-1
# baseline (focal loss, oversampling off). The published Part-2 numbers came
# from the older two-model ensemble, which Part 1 has since superseded.
# Inference only, no training.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/41_cross_device_new_baseline.py
