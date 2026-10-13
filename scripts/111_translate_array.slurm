#!/bin/bash
#SBATCH --job-name=xlate
#SBATCH --account=trends517s113
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/xlate_%x_%A_%a.out
# One shard of the full BRSET-TRAIN translation. Resumable: images already
# written are skipped, so a preempted/requeued shard just continues.
set -euo pipefail
: "${RUN:?}"; : "${CKPT:?}"; : "${OUT_TAG:?}"; : "${N_SHARDS:?}"
: "${INIT:=physics}"; : "${STRENGTH:=0.4}"; : "${STEPS:=30}"; : "${W_STRUCT:=1.5}"
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/111_translate.py --run "$RUN" --ckpt "$CKPT" \
  --out_tag "$OUT_TAG" --pool train --subset all --init "$INIT" --strength "$STRENGTH" --steps "$STEPS" \
  --w_struct "$W_STRUCT" --shard "$SLURM_ARRAY_TASK_ID" --n_shards "$N_SHARDS"
