#!/bin/bash
#SBATCH --job-name=forensics
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/forensics_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/forensics_%j.err
#
# MODEL FORENSICS: where and how the BRSET->mBRSET classifier fails.
#
# CPU ONLY, NO GPU. Reads nothing but cached prediction vectors
# (results/*/test_predictions.npz, results/*/val_predictions.npz) and the
# ordered test label table. No image decoding, no model loading, no training.
# Deliberately on qCPU24 so it never competes with the three authorised GPU
# jobs (AGENTS.md: numerical tests run on allocated nodes, not the login node).
#
# What it answers, in order:
#   1  ranking versus calibration: oracle-cutpoint F1 vs val-tuned F1
#   2  which positives are missed, and the hard core no model recovers
#   3  score-distribution shift versus genuine overlap
#   4  cross-tab of what real target data fixes against what synthesis fixes
#   5  agreement structure across arms (Spearman, error-set McNemar)
#   6  exhaustive ensemble search over the zero-shot-eligible arms
#   7  how concentrated the pooled model's AUC advantage is
#   8  cross-links between those sets, plus the label-efficiency curve
#
# Bars for calling anything an effect (seed-only noise floors):
#   DR AUC 0.0142, DR F1 0.0199.
#
# Output: results/model_forensics.json (aggregates and row indices only; no
# filenames, no patient identifiers, per the data use agreement).
#
# Usage:
#   sbatch scripts/93_model_forensics.slurm
#   sbatch --export=ALL,SECTIONS=1,6 scripts/93_model_forensics.slurm
#   sbatch --export=ALL,LABEL_INDEX=1,OUT=results/model_forensics_me.json \
#          scripts/93_model_forensics.slurm

set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

: "${SECTIONS:=all}"
: "${LABEL_INDEX:=0}"
: "${OUT:=results/model_forensics.json}"
: "${NBOOT:=2000}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== 93_model_forensics on $(hostname) at $(date -u) ==="
echo "sections=${SECTIONS} label_index=${LABEL_INDEX} out=${OUT} nboot=${NBOOT}"

"$PY" scripts/93_model_forensics.py \
  --results-dir /home/users/sthummala2/brset-convnextv2/results \
  --labels-csv /home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel/test/labels.csv \
  --out "/home/users/sthummala2/brset-convnextv2/${OUT}" \
  --label-index "${LABEL_INDEX}" \
  --sections "${SECTIONS}" \
  --n-boot-ci "${NBOOT}"

echo "=== done at $(date -u) ==="
