#!/bin/bash
#SBATCH --job-name=audit
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/audit_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/audit_%j.err
# CPU only. Touches no GPU.
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python scripts/74_audit_physics.py
