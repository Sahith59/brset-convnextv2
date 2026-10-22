#!/bin/bash
#SBATCH --job-name=c1fs
#SBATCH --account=trends517s113
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/c1fs_%x_%j.out
# D103 frequency-split twins. STAGE=build (CPU): pilot -> sigma -> all 11,372 images + zoom montage.
#                              STAGE=gates (GPU): copy check vs mBRSET reference -> data/syn_c1fs.
# Classifiers: scripts/139_synth_data_cls.slurm with DATA=syn_c1fs ALLOW=<synth>/full_c1fs (afterok on gates).
set -euo pipefail
: "${STAGE:?}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
G=/data/users3/sthummala2/brset_gen
cd /home/users/sthummala2/brset-convnextv2
if [ "$STAGE" = build ]; then
  "$PY" -u scripts/146_freqsplit.py --mode pilot --workers "${SLURM_CPUS_PER_TASK:-16}"
  SIG=$("$PY" -c "import json; print(json.load(open('results/freqsplit_pilot.json'))['chosen_sigma'])")
  rm -rf "$G/synth/full_c1fs"
  "$PY" -u scripts/146_freqsplit.py --mode build --sigma "$SIG" --workers "${SLURM_CPUS_PER_TASK:-16}"
  n=$(ls "$G/synth/full_c1fs" | grep -c '\.png$'); echo "images: $n"; [ "$n" -eq 11372 ]
  "$PY" - <<'PYEOF'
# visual check: 4 photos x (C1 twin | C1FS twin) at full resolution, 200-px crop near the centre of the retina
import numpy as np, pandas as pd
from PIL import Image
G = "/data/users3/sthummala2/brset_gen"
meta = pd.read_csv(f"{G}/cache/brset_train_meta.csv")
pick = meta[meta.grade.isin([1, 2])].sample(4, random_state=3)
rows = []
for f in pick.file:
    st = f.split(".")[0]
    a = np.asarray(Image.open(f"{G}/synth/full_gen_both_dc_phys_s0.4_w1.5/{st}.png").convert("RGB"))[180:380, 180:380]
    b = np.asarray(Image.open(f"{G}/synth/full_c1fs/{st}.png").convert("RGB"))[180:380, 180:380]
    rows.append(np.concatenate([a, np.full((200, 6, 3), 255, np.uint8), b], 1))
Image.fromarray(np.concatenate(rows, 0)).resize((812, 800), Image.NEAREST).save(f"{G}/analysis/c1fs_zoom.png")
print("montage ->", f"{G}/analysis/c1fs_zoom.png", "grades", pick.grade.tolist())
PYEOF
elif [ "$STAGE" = gates ]; then
  "$PY" -u scripts/132_copy_check.py --tag c1fs --ref mbrset --n 1000 --synth_dir "$G/synth/full_c1fs"
  "$PY" - <<'PYEOF'
import json, sys
d = json.load(open("results/copycheck_c1fs.json"))
r, ref = d["sets"]["full_c1fs"]["copy_rate"], d["reference_val"]["or_flag_rate"]
print(f"D103 copy gate: flagged {r:.4f} vs new-patient OR-rate {ref:.4f} ->", "PASS" if r <= ref else "FAIL")
sys.exit(0 if r <= ref else 3)
PYEOF
  "$PY" -u scripts/133_build_synth_data.py --name syn_c1fs --twins "$G/synth/full_c1fs"
else
  echo "unknown STAGE $STAGE"; exit 2
fi
