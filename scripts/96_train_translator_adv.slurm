#!/bin/bash
#SBATCH --job-name=xadv
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/xadv_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/xadv_%j.err
#
# TRANSLATOR WITH THE ADVERSARIAL TERM ON. Diagnosis-driven.
#
# The statistics-only translator failed downstream at every mixing ratio
# (-0.0060 to -0.0452 DR AUC). Two measurements on 60 images explain why:
#
#   H1  red-lesion candidate retention 0.416, spurious 0.668
#       It destroys 58 percent of lesion evidence and two thirds of the
#       candidates in its output are fabricated. GDRNet, which we criticise,
#       retains 0.465. Ours was worse.
#
#   H2  a logistic probe on frozen ImageNet features separates real mBRSET
#       from UNTOUCHED BRSET at AUC 1.0000, and from SYNTHETIC BRSET at AUC
#       1.0000. Change: +0.0000.
#
# H2 is the decisive one. Matching 13 appearance statistics to |z| <= 0.32 moved
# the images ZERO distance in the feature space a network actually uses. We
# optimised the wrong objective, and we did it with the only loss that directly
# penalises separability switched off: --w_adv defaulted to 0.0.
#
# A discriminator penalises exactly the quantity H2 measures. This run turns it
# on. Two weights, since adversarial scale is the one hyperparameter that
# matters here.
#
# GATE BEFORE SPENDING DOWNSTREAM GPU: rerun scripts/90_diagnose_translator.py
# on the result. If probe AUC does not fall materially below 1.0000, the
# translator still is not moving the images and no downstream run is justified.
# H2 costs minutes; a downstream run costs five GPU-hours.
set -euo pipefail
: "${WADV:=1.0}"
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/83_train_translator.py \
  --targets results/chromatic_stats_sq512.json \
  --presize 560 --size 512 \
  --eps_lf 0.15 --eps_hf 0.0 --hf_mode off \
  --res_mode l2 --max_artifact 0.02 \
  --w_adv "${WADV}" \
  --steps 4000 --batch_size 8 \
  --amp bf16 --seed 0 \
  --save_panel --eval_every 500 \
  --task "xlate_adv${WADV}" \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
