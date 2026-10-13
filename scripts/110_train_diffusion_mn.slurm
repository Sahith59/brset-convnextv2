#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gen_%x_%j.out
#
# MULTI-NODE training of the stage-2 diffusion renderer. Pass --nodes=N to sbatch;
# each node contributes 2 A40s, one rank per GPU, 16 images per rank.
#   --nodes=4  ->  8 GPUs, effective batch 128
# More GPUs do not shorten a step; they enlarge the batch, so the schedule is
# restated in steps of the larger batch (LR raised, fewer steps, same images seen).
#
# Each rank is its own srun task, so the launcher sets the rank variables that
# torch.distributed reads. NCCL_P2P_DISABLE=1 is required on these nodes (the
# direct GPU-to-GPU path hangs); see the single-node script for the measurement.
set -euo pipefail
: "${TAG:?}"; : "${DOMAINS:=target}"; : "${COND_BLUR:=1.8}"; : "${STEPS:=40000}"; : "${LR:=2e-4}"
: "${WARMUP:=1000}"; : "${CACHE_TAG:=}"; : "${CKPT_EVERY:=1000}"; : "${KEEP_EVERY:=2500}"
: "${SAMPLE_EVERY:=2500}"; : "${LOG_EVERY:=100}"
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export TAG DOMAINS COND_BLUR STEPS LR WARMUP CACHE_TAG CKPT_EVERY KEEP_EVERY SAMPLE_EVERY LOG_EVERY
cd /home/users/sthummala2/brset-convnextv2
echo "nodes: $SLURM_JOB_NODELIST  tasks: $SLURM_NTASKS  master: $MASTER_ADDR:$MASTER_PORT  backend: ${GEN_BACKEND:-nccl}"
srun --ntasks="$SLURM_NTASKS" --ntasks-per-node=2 --cpus-per-task=10 bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  exec /home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/110_train_diffusion.py \
    --tag "$TAG" --domains "$DOMAINS" --cond_blur "$COND_BLUR" --steps "$STEPS" --lr "$LR" \
    --warmup "$WARMUP" --cache_tag "$CACHE_TAG" --batch 16 --workers 6 --log_every "$LOG_EVERY" \
    --ckpt_every "$CKPT_EVERY" --keep_every "$KEEP_EVERY" --sample_every "$SAMPLE_EVERY"'
