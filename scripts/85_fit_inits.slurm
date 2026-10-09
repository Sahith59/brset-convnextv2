#!/bin/bash
#SBATCH --job-name=fitinit
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-3
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/fitinit_%A_%a.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/fitinit_%A_%a.err
#
# CPU only. Does not touch the eight running GPU jobs or the three-GPU
# authorisation. Measured cost on a login-node core count: 9.6 s per step for
# 64 images at 512x512, so 400 steps is roughly one hour per arm.
#
# IDENTIFIABILITY, not replication. Task 0 starts at the FITTED values and is
# the headline fit. Tasks 1 to 3 are random restarts drawn uniformly inside the
# bounds. If a parameter lands in a different place at comparable loss then it
# is not identified by these statistics, and no claim may be made about it,
# including the claim that it pins at a bound and the model is therefore
# exhausted. The source images and the target file are identical across arms,
# so the only difference is where the optimiser starts.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
TGT=results/chromatic_stats_sq512.json
if [ ! -f "$TGT" ]; then
  echo "missing $TGT: run scripts/85_measure_sq512.slurm first" >&2
  exit 1
fi
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  SEED=-1
  OUT=results/physics_v2_fit.json
else
  SEED="${SLURM_ARRAY_TASK_ID}"
  OUT="results/physics_v2_fit_init${SLURM_ARRAY_TASK_ID}.json"
fi
echo "arm ${SLURM_ARRAY_TASK_ID}: init_seed ${SEED} -> ${OUT}"
"$PY" scripts/79_fit_physics_v2.py --targets "$TGT" \
      --presize 560 --size 512 --steps 400 \
      --init_seed "${SEED}" --out "${OUT}"
