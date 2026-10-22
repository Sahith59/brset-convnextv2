#!/bin/bash
#SBATCH --job-name=painter
#SBATCH --account=trends517s113
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/painter_%x_%j.out
#
# D097 C6 lesion painter (scripts/141_train_painter.py). Partition, nodes, GPUs from the sbatch line,
# one rank per GPU. Requeue-safe: resumes from runs/<TAG>/ckpt_last.pt.
set -euo pipefail
: "${TAG:?}"; : "${EXTRA:=}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} OMP_NUM_THREADS=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TAG EXTRA
cd /home/users/sthummala2/brset-convnextv2
echo "nodes: $SLURM_JOB_NODELIST  tasks: $SLURM_NTASKS  extra: $EXTRA"
srun --ntasks="$SLURM_NTASKS" bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  exec /home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/141_train_painter.py --tag "$TAG" $EXTRA'
