#!/bin/bash
#SBATCH --job-name=xlate
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/xlate_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/xlate_%j.err
#
# Train the two-stage BRSET -> mBRSET translator. Stage 1 is the bounded
# differentiable acquisition model, stage 2 the bounded residual refiner.
#
# eps_hf is set BELOW the default. The self-test reports a worst-case stochastic
# adjacent-pixel edit of 0.0400 against a measured lesion contrast floor of 0.04,
# i.e. exactly at the amplitude that can erase a microaneurysm. Measured lesion
# retention confirms the risk is real: under the noise sweep, spurious candidate
# rate reaches 0.92 to 0.94, meaning over nine in ten detected lesions in the
# synthetic image are fabricated. The refiner is therefore held well under the
# lesion floor and the high-frequency branch is disabled in this first run.
#
# res_mode l2: the self-test refuted the l1_down rationale (it equals l1_full on
# a one-signed residual, ratio 6.67), and a dense low-frequency chroma shift is
# exactly what we need the refiner to produce. L1 is a sparsity prior and would
# charge most for the thing we want.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" scripts/83_train_translator.py \
  --targets results/chromatic_stats_sq512.json \
  --presize 560 --size 512 \
  --eps_lf 0.15 --eps_hf 0.0 --hf_mode off \
  --res_mode l2 \
  --steps 4000 --batch_size 8 \
  --amp bf16 --seed 0 \
  --save_panel --eval_every 250 \
  --max_artifact 0.02 \
  --task xlate_v2_noartifact \
  --output_dir /home/users/sthummala2/brset-convnextv2/results
