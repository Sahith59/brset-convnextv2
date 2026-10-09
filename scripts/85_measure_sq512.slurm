#!/bin/bash
#SBATCH --job-name=chrsq512
#SBATCH --account=trends517s113
#SBATCH --partition=qCPU24
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/chrsq512_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/chrsq512_%j.err
#
# CPU only, no GPU. Image decoding must not run on a login node (AGENTS.md).
#
# Re-measures the fit targets at the resolution synthesis is actually applied
# at. The native-resolution targets in results/chromatic_stats_native.json were
# measured with a 2491 px BRSET disc and a 1599 px mBRSET disc, while the fit
# operates on 512 px tensors, and Laplacian variance is strongly scale
# dependent (BRSET 0.000078 native against 0.000257 squashed to 512). Fitting
# 512 px tensors to native targets therefore asked the model to close about a
# quarter of the real sharpness gap, in the direction that makes its own
# noise-share audit pass.
#
# presize 560 then crop 512 is the classifier's transform. The trainer uses
# RandomCrop and this uses CenterCrop; they differ by a translation only, which
# does not change the sampling pitch and so does not change any statistic here
# in expectation.
#
# Adds per-channel Laplacian variance, without which the fit has no target that
# can see the channel-asymmetric high-frequency excess the model exists to
# reproduce.
set -euo pipefail
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
echo "### trainer pitch: Resize((560,560)) then 512 centre crop  [FIT TARGETS]"
"$PY" scripts/76_measure_chromatic.py --n 400 --presize 560 --crop 512 \
      --out results/chromatic_stats_sq512.json
echo
echo "### native resolution, re-run so the per-channel lapvar columns exist there too"
echo "### (camera comparison only, NOT a fit target). Written to a NEW file: the"
echo "### 9 Oct native measurement is an experiment artifact and is not overwritten."
"$PY" scripts/76_measure_chromatic.py --n 400 \
      --out results/chromatic_stats_native_lapvar.json
