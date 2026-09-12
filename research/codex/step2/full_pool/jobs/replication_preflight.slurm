#!/bin/bash
#SBATCH --job-name=br_rep_preflight
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/replication_preflight_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/research/codex/step2/full_pool/logs/replication_preflight_%j.err
set -euo pipefail
cd /home/users/sthummala2/brset-convnextv2
python -m py_compile \
  research/codex/step2/full_pool/frozen/assess_replication.py \
  research/codex/step2/full_pool/frozen/replication_preflight.py
bash -n \
  research/codex/step2/full_pool/jobs/init_seed.slurm \
  research/codex/step2/full_pool/jobs/train_seed_wave.slurm \
  research/codex/step2/full_pool/jobs/assess_seed.slurm
/home/users/sthummala2/.conda/envs/retfound/bin/python \
  research/codex/step2/full_pool/frozen/replication_preflight.py
