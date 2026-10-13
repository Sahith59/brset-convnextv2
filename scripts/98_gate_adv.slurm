#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gate_%j.out
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
# The gate: statistics-only translation left the domain probe at AUC 1.0000,
# i.e. it moved the images zero distance in feature space. A downstream run
# costs 5 GPU-hours; this check costs minutes. No translator earns a downstream
# run unless probe AUC falls materially below 1.0.
for W in 0.5 2.0; do
  echo "############ w_adv=${W}"
  "$PY" -u scripts/90_diagnose_translator.py --n 60 \
      --ckpt "results/xlate_adv${W}/translator_last.pth" \
      --out "results/translator_diagnosis_adv${W}.json"
done
