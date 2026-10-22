#!/bin/bash
#SBATCH --job-name=B_cls
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/B_cls_%x_%j.out
# D105a experiment B (X-mild / X-nat): scripts/139 (C1-prev's recipe) byte-for-byte PLUS
# --extra_manifest results/extra_manifest.csv --extra_n 0, which adds NO rows and turns on the oedema mask for the
# domain-2 (translated external) rows of data/<DATA>/train/labels.csv (built by scripts/152).
set -euo pipefail
: "${DATA:?}"; : "${SEED:?}"; : "${TASK:?}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
# HAND-OFF (move a run to a faster GPU): stop the old job, confirm it is gone, clear its output
# folder so no file from the old run survives, then start from scratch. Two jobs never share a folder.
if [ -n "${CANCEL_FIRST:-}" ]; then
  scancel "$CANCEL_FIRST" 2>/dev/null || true
  for i in $(seq 1 36); do squeue -h -j "$CANCEL_FIRST" 2>/dev/null | grep -q . || break; sleep 5; done
  if squeue -h -j "$CANCEL_FIRST" 2>/dev/null | grep -q .; then echo "job $CANCEL_FIRST still alive; refusing"; exit 4; fi
  OLD="/data/users3/sthummala2/brset_gen/cls_runs/${TASK}"
  [ -n "${TASK}" ] && [ -d "$OLD" ] && rm -rf "$OLD" && echo "hand-off: cleared $OLD"
  sleep 10
fi
"$PY" - <<PYEOF
import pandas as pd
from pathlib import Path
df = pd.read_csv("data/${DATA}/train/labels.csv")
bad = [f for f in df.file if not str(Path(f).resolve()).startswith("/data/users3/sthummala2/brset_gen/synth/")]
assert not bad, f"{len(bad)} training images are not synthetic, e.g. {bad[:3]}"
allow = [a.rstrip("/") + "/" for a in "${ALLOW:-}".split(",") if a.strip()]
if allow:   # D102: explicit allow-list of synthetic folders (the synth tree also holds invalid near-copy folders)
    bad = [f for f in df.file if not any(str(Path(f).resolve()).startswith(a) for a in allow)]
    assert not bad, f"{len(bad)} training images outside the allow-list, e.g. {bad[:3]}"
    print(f"audit: all training images inside the allow-list {allow}")
print(f"audit: {len(df)} training images, all under brset_gen/synth; DR+ {int(df.diabetic_retinopathy.sum())}")
PYEOF
echo "=== ${TASK}: synthetic-only data ${DATA} seed=${SEED} ==="
"$PY" scripts/30_train_strong_baseline.py \
  --model convnextv2_large.fcmae_ft_in22k_in1k_384 \
  --data_path /home/users/sthummala2/brset-convnextv2/data/${DATA} \
  --nb_classes 2 --input_size 512 --resize_size 560 \
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3 \
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2 \
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200 \
  --ckpt_freq 0 --num_workers 8 --seed "${SEED}" --no_oversample --amp_dtype bf16 \
  --loss focal --focal_gamma 2.0 \
  --extra_manifest /home/users/sthummala2/brset-convnextv2/results/extra_manifest.csv --extra_n 0 \
  --task "${TASK}" \
  --output_dir /data/users3/sthummala2/brset_gen/cls_runs
