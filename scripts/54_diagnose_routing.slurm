#!/bin/bash
#SBATCH --job-name=diag_route
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/diag_route_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/diag_route_%j.err
#
# Measures the conditional-entropy difference that drives condent routing, to
# decide whether alpha sitting at 0.501 is a finding or a temperature bug.
# Inference only, no training.

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/53_diagnose_routing_signal.py
