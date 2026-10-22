#!/bin/bash
#SBATCH --job-name=pilot
#SBATCH --account=trends517s113
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/pilot_%x_%A_%a.out
# D091 pilot: 256 cascade images per setting. SETTINGS = space-separated "w_class:strength" list,
# one per array task. Output synth/pilot_<BASE_TAG>_w<w>_s<strength>.
set -euo pipefail
: "${BASE_RUN:?}"; : "${BASE_CKPT:?}"; : "${BASE_TAG:?}"; : "${SETTINGS:?}"
: "${RENDER_RUN:=gen_both_dc}"; : "${RENDER_CKPT:=ema_0002000.pt}"; : "${LIMIT:=256}"
read -ra S <<< "$SETTINGS"; ONE=${S[$SLURM_ARRAY_TASK_ID]}; W=${ONE%%:*}; ST=${ONE##*:}
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/131_sample_cascade.py --base_run "$BASE_RUN" --base_ckpt "$BASE_CKPT" \
  --render_run "$RENDER_RUN" --render_ckpt "$RENDER_CKPT" --out_tag "pilot_${BASE_TAG}_w${W}_s${ST}" \
  --w_class "$W" --strength "$ST" --limit "$LIMIT" --batch 8
