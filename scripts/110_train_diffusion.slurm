#!/bin/bash
#SBATCH --job-name=gen
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=4-00:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/gen_%x_%j.out
#
# STAGE 2, FOR REAL: a ~105M-parameter conditional diffusion renderer
# p(image | structure map, device), replacing the 68k-parameter bounded refiner
# that could not leave a neighbourhood of its input.
#
# Dr. Ye, 8 Oct: "image synthesis converting BRSET to synthesized mBRSET ... joint
# training ... make use of the generative world model". One retina, two device
# renderings: the structure map is the device-invariant state that carries the
# label, the diffusion model is the learned renderer of the target device.
#
# Two A40s, batch 16 each (the model needs 37.5 GB at batch 16). Resumable: a
# resubmission picks up from runs/<TAG>/ckpt_last.pt. Never reads an mBRSET
# label or any val/test image.
#
#   TAG=gen_tgt        DOMAINS=target  COND_BLUR=1.8   target-only renderer
#   TAG=gen_both       DOMAINS=both    COND_BLUR=1.8   device-conditioned world model
#   TAG=gen_tgt_sharp  DOMAINS=target  COND_BLUR=0.8   tighter lesion pinning
set -euo pipefail
: "${TAG:?}"; : "${DOMAINS:=target}"; : "${COND_BLUR:=1.8}"; : "${STEPS:=120000}"
: "${CACHE_TAG:=}"; : "${CKPT_EVERY:=2500}"; : "${KEEP_EVERY:=10000}"; : "${SAMPLE_EVERY:=5000}"
# NCCL's direct GPU-to-GPU path (P2P over CUDA IPC) HANGS on the arctrdagn nodes:
# both ranks stall in DDP's first collective (_verify_param_shape_across_processes)
# at 100% CPU with the model loaded and no step ever run. Disabling P2P makes NCCL
# use shared memory instead, which works. NCCL_IB_DISABLE alone does NOT fix it.
# Measured 12 Oct 2026 with four parallel trials.
export NCCL_P2P_DISABLE=1
PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2
"$PY" -u -m torch.distributed.run --nproc_per_node=2 scripts/110_train_diffusion.py \
  --tag "$TAG" --domains "$DOMAINS" --cond_blur "$COND_BLUR" --steps "$STEPS" \
  --cache_tag "$CACHE_TAG" --batch 16 --workers 8 \
  --ckpt_every "$CKPT_EVERY" --keep_every "$KEEP_EVERY" --sample_every "$SAMPLE_EVERY"
