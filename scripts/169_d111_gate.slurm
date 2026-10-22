#!/bin/bash
#SBATCH --job-name=d111gate
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/d111gate_%x_%j.out
# D111 smoke follow-up: does the renderer's camera token matter? Same gate subset (546 held-out BRSET), sampler and checkpoint
# as the Table-1 renders (gen_both ema_0040000, 30 steps, t0 0.4, w 1.5, seed 0); only INIT and TARGET vary. Then gates 1-2
# (camera classifier P(phone), judge label preservation) exactly as scripts/112_gate.slurm.
set -euo pipefail
: "${INIT:?}"; : "${TARGET:?}"; : "${OUT_TAG:?}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" -u scripts/111_translate.py --run gen_both --ckpt ema_0040000.pt --out_tag "$OUT_TAG" --subset gate --pool heldout \
  --init "$INIT" --strength 0.4 --steps 30 --w_struct 1.5 --w_device 0.0 --target "$TARGET"
"$PY" -u scripts/112_gates.py --synth_tag "$OUT_TAG"
