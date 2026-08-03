#!/bin/bash
#SBATCH --job-name=moe_gate
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/moe_gate_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/moe_gate_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2
python3 scripts/27_moe_gate.py
