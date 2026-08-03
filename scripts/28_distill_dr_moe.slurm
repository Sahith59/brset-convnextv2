#!/bin/bash
#SBATCH --job-name=distill_dr_moe
#SBATCH --account=trends517s113
#SBATCH --partition=qGPU24
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/distill_dr_moe_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/distill_dr_moe_%j.err

source /etc/profile.d/Z01-modules.sh
module load miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate retfound

cd /home/users/sthummala2/brset-convnextv2
python3 scripts/28_distill_dr_moe.py
