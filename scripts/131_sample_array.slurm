#!/bin/bash
#SBATCH --job-name=cascade
#SBATCH --account=trends517s113
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/cascade_%x_%A_%a.out
# One shard of the full D091 cascade set (resumable).
set -euo pipefail
: "${BASE_RUN:?}"; : "${BASE_CKPT:?}"; : "${OUT_TAG:?}"; : "${N_SHARDS:?}"; : "${W:?}"; : "${ST:?}"
: "${RENDER_RUN:=gen_both_dc}"; : "${RENDER_CKPT:=ema_0002000.pt}"; : "${MULT:=3}"
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/131_sample_cascade.py --base_run "$BASE_RUN" --base_ckpt "$BASE_CKPT" \
  --render_run "$RENDER_RUN" --render_ckpt "$RENDER_CKPT" --out_tag "$OUT_TAG" --mult "$MULT" \
  --w_class "$W" --strength "$ST" --shard "$SLURM_ARRAY_TASK_ID" --n_shards "$N_SHARDS" --batch 8
