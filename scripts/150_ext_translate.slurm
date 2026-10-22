#!/bin/bash
#SBATCH --job-name=B_xlate
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPUM
#SBATCH --gres=gpu:L40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/B_xlate_%A_%a.out
# D105 B: translate the external (DDR '007' + Messidor-2) cache with C1's EXACT recipe (as logs/xlate_xl_dc_4628564_*):
# physics xlate_v2_noartifact -> runs/gen_both_dc/ema_0002000.pt, strength 0.4, 30 steps, w_struct 1.5, w_device 0, seed 0.
# Reads brset_gen/cache/ext2_brset_train_560.npy (+ _meta.csv) built by scripts/149_ext_build.py. Resumable.
set -euo pipefail
: "${N_SHARDS:?}"
cd /home/users/sthummala2/brset-convnextv2
/home/users/sthummala2/.conda/envs/retfound/bin/python -u scripts/111_translate.py --run gen_both_dc --ckpt ema_0002000.pt \
  --out_tag ext2_dc_matfix_s0.4_w1.5 --cache_tag ext2_ --source brset --pool train --subset all --init physics \
  --strength 0.4 --steps 30 --w_struct 1.5 --w_device 0.0 --seed 0 --shard "$SLURM_ARRAY_TASK_ID" --n_shards "$N_SHARDS"
