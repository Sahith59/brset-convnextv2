#!/bin/bash
#SBATCH --job-name=gate
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gate_%x_%j.out
# Translate the stratified gate subset with one generator snapshot + one sampler
# setting, then run gates 1 and 2 on it. Cheap by design: this is what decides
# whether a setting earns a full translation and a downstream run.
set -euo pipefail
: "${RUN:?}"; : "${CKPT:=ckpt_last.pt}"; : "${OUT_TAG:?}"
: "${INIT:=physics}"; : "${STRENGTH:=0.6}"; : "${STEPS:=50}"; : "${W_STRUCT:=1.5}"; : "${W_DEVICE:=0.0}"; : "${LIMIT:=0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
# the job is timed to start just after its snapshot should exist; if training was
# slowed or requeued, wait for the file (up to 50 min) instead of failing
F=/data/users3/sthummala2/brset_gen/runs/$RUN/$CKPT
for i in $(seq 1 "${WAIT_TRIES:-100}"); do [ -f "$F" ] && break; sleep 30; done
[ -f "$F" ] || { echo "snapshot $F never appeared"; exit 3; }
sleep 20   # let the write finish
"$PY" -u scripts/111_translate.py --run "$RUN" --ckpt "$CKPT" --out_tag "$OUT_TAG" --subset gate \
  --pool heldout --init "$INIT" --strength "$STRENGTH" --steps "$STEPS" --w_struct "$W_STRUCT" --w_device "$W_DEVICE" --limit "$LIMIT"
"$PY" -u scripts/112_gates.py --synth_tag "$OUT_TAG"
