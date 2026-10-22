#!/bin/bash
#SBATCH --job-name=forensic
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/forensic_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/forensic_%j.err
#
# CPU-only. Uses no GPU, so it does not touch the three authorised training
# jobs or the three-concurrent-GPU limit.
#
# Why this runs on a compute node rather than the login node: --image_stats
# decodes the 732 test JPEGs, and the AGENTS.md rule puts all image decoding on
# allocated compute. The metadata half is cheap, but there is no reason to split
# the run across two machines.
#
# What it answers: the BRSET->mBRSET DR AUC gap (0.9104 zero-shot vs 0.9463
# pooled) survives after class balance and image appearance were both ruled out
# experimentally. This measures the structural, label-level and population-level
# differences from the ORIGINAL PhysioNet metadata, which carries ICDR severity
# grades, quality flags, laterality and demographics that the training pipeline
# throws away.
#
# Output: results/data_forensics.json (aggregates only, per the DUA).
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/92_data_forensics.py \
  --image_stats \
  --out /home/users/sthummala2/brset-convnextv2/results/data_forensics.json
