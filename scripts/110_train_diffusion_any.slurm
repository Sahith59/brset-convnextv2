#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --account=trends517s113
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gen_%x_%j.out
#
# Hardware-agnostic launcher for the diffusion renderer: partition, nodes, tasks
# and GPUs all come from the sbatch command line, one rank per GPU. Measured
# images/s per GPU for this 105M UNet at 256 px (12 Oct 2026):
#     A40 10.5  (batch 16, 37.5 GB)      A100-40GB 22.4  (batch 12 is the safe fit)
#     H100 47.3 (batch 24 safe; 32 peaks at 74 of 79 GB)
# The fast partitions preempt by REQUEUE; a requeued job resumes from
# runs/<TAG>/ckpt_last.pt, so at most CKPT_EVERY steps are lost.
set -euo pipefail
: "${TAG:?}"; : "${BATCH:?}"; : "${DOMAINS:=target}"; : "${COND:=structure}"; : "${COND_BLUR:=1.8}"; : "${STEPS:=40000}"
: "${LR:=2e-4}"; : "${WARMUP:=1000}"; : "${CKPT_EVERY:=500}"; : "${KEEP_EVERY:=2500}"
: "${SAMPLE_EVERY:=2500}"; : "${LOG_EVERY:=100}"; : "${LR_DECAY_START:=-1}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} OMP_NUM_THREADS=4
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TAG BATCH DOMAINS COND COND_BLUR STEPS LR WARMUP CKPT_EVERY KEEP_EVERY SAMPLE_EVERY LOG_EVERY LR_DECAY_START
cd /home/users/sthummala2/brset-convnextv2
# HAND-OFF: take over a run from a slower job. Two jobs must never write one run
# directory, so the old job is cancelled and confirmed gone before this one starts.
if [ -n "${CANCEL_FIRST:-}" ]; then
  scancel "$CANCEL_FIRST" 2>/dev/null || true
  for i in $(seq 1 36); do squeue -h -j "$CANCEL_FIRST" 2>/dev/null | grep -q . || break; sleep 5; done
  if squeue -h -j "$CANCEL_FIRST" 2>/dev/null | grep -q .; then
    echo "job $CANCEL_FIRST is still alive; refusing to share its run directory"; exit 4
  fi
  sleep 20
fi
echo "nodes: $SLURM_JOB_NODELIST  tasks: $SLURM_NTASKS  batch/GPU: $BATCH  master: $MASTER_ADDR:$MASTER_PORT"
srun --ntasks="$SLURM_NTASKS" bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  exec /home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/110_train_diffusion.py \
    --tag "$TAG" --domains "$DOMAINS" --cond "$COND" --cond_blur "$COND_BLUR" --steps "$STEPS" --lr "$LR" \
    --warmup "$WARMUP" --batch "$BATCH" --workers 6 --log_every "$LOG_EVERY" \
    --ckpt_every "$CKPT_EVERY" --keep_every "$KEEP_EVERY" --sample_every "$SAMPLE_EVERY" --lr_decay_start "$LR_DECAY_START"'
