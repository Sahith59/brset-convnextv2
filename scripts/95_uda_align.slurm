#!/bin/bash
#SBATCH --job-name=udaalign
#SBATCH --account=trends517s113
#SBATCH --partition=qTRDGPU
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/home/users/sthummala2/brset-convnextv2/logs/udaalign_%j.out
#SBATCH --error=/home/users/sthummala2/brset-convnextv2/logs/udaalign_%j.err
#
# UNSUPERVISED DOMAIN ADAPTATION ON UNLABELED mBRSET IMAGES
# =========================================================
# NOT SUBMITTED by the agent that wrote it: three GPU jobs were already running
# and the authorised limit is three. Submit by hand when a slot frees.
#
# THE METHOD. Identical to the budget-0 label-efficiency baseline in every
# respect (same backbone, resolution, loss, sampler, LR, epochs, EMA,
# thresholding and mBRSET val/test protocol) plus ONE extra term. Each step
# draws a labelled BRSET batch and an UNLABELED mBRSET batch, computes the
# supervised loss on BRSET only, and adds
#
#     lambda * L_align( pooled features of the two batches )
#
# with L_align either Deep CORAL (covariance matching, Sun & Saenko ECCV 2016W)
# or multi-scale RBF MMD (the DAN form, Long et al. ICML 2015). No
# discriminator, no gradient reversal, no adversarial stability tricks.
#
# WHY THIS ARM EXISTS. data/finetune_mbrset_multilabel/train holds 3,402 real
# target-domain images that no arm has yet used WITHOUT labels. Zero-shot
# ignores them; pooling consumes them with labels. In the label-cost framing
# this arm still spends zero target TRAINING labels, so it belongs at x=0 on
# the label-efficiency curve.
#
# NUMBERS IT MUST BEAT (DR, mBRSET test, 732 images, val-fitted thresholds):
#   zero-shot / budget 0, seed 0   AUC 0.9104   F1 0.7774   (results/lebase_b0_s0)
#     seeds 1 and 2                AUC 0.8906 / 0.9009      F1 0.7559 / 0.7677
#   pooling, all 3,402 LABELLED    AUC 0.9463   F1 0.8365   <- the ceiling
#   250 labelled target images     AUC 0.9278   F1 0.7785   (results/lebase_b250_s0)
#   seed noise floor               AUC 0.0142   F1 0.0199
# A result is only interesting if it clears 0.9104 + 0.0142 = 0.9246 AUC, i.e.
# if unlabeled target data buys at least what ~250 labelled target images buy.
# One seed proves nothing here: five previous architectures all landed inside
# the noise floor. Budget three seeds (0, 1, 2) before believing anything.
#
# KNOWN THEORETICAL WEAKNESS, stated up front. BRSET train is 6.6% DR-positive
# and mBRSET is 23.4%. Marginal feature alignment forces p_s(f) towards p_t(f)
# while the label priors differ by 3.5x, which is exactly the regime where
# Zhao et al. (ICML 2019) show invariant representations can RAISE target
# error. Two mitigations exist, both as separate arms with their own controls:
#   * ARM C below matches the source positive rate to the target's 23.4% with
#     the existing --target_pos_rate. That scalar comes from target labels, so
#     it is a (small, disclosed) use of target label information and needs its
#     own no-alignment control at the same sampler setting.
#   * estimating the target prior from the unlabeled set (BBSE/EM) instead.
#
# USAGE
#   sbatch --export=ALL,ARM=mmd,SEED=0,LAMBDA=0.1 scripts/95_uda_align.slurm
#   sbatch --export=ALL,ARM=coral,SEED=0 scripts/95_uda_align.slurm
#   sbatch --export=ALL,ARM=stage,SEED=0 scripts/95_uda_align.slurm
#   sbatch --export=ALL,ARM=prior,SEED=0 scripts/95_uda_align.slurm
#   sbatch --export=ALL,ARM=st,SEED=0    scripts/95_uda_align.slurm
#
# ORDER OF PRIORITY if only one GPU is free: ARM=mmd. The measured gap is that
# mBRSET is noisier, higher-contrast and DIMMER, i.e. partly a first-order
# shift, and published Deep CORAL matches second moments only and is provably
# blind to a pure mean shift (unit test 1). RBF-kernel MMD sees every moment.
#
# TUNING THE WEIGHT. There is no way to guess a literal lambda here: Deep
# CORAL carries a 1/(4 d^2) normalisation and MMD does not, so their raw values
# differ by orders of magnitude and depend on the feature width. Every arm below
# therefore uses --uda_scale_mode relative, which measures sup/align once at the
# first step and folds it in, so --uda_weight 0.1 means "the alignment term
# starts at 10 percent of the supervised loss" for either criterion. The
# measured scale is printed once ("[uda] scale calibration: ...") and logged in
# results/<task>/uda_log.jsonl, so the arm can be re-quoted later as a literal
# weight with --uda_scale_mode absolute.
#
# WATCH THE FIRST TWO EPOCHS AND KILL THE JOB RATHER THAN SPEND 12 HOURS ON IT:
#   * val macro_auc collapsing from epoch 0 -> lambda too high; the encoder is
#     being paid to throw information away. Halve --uda_weight.
#   * mean_align flat from epoch 0 to 3 and val identical to lebase_b0 ->
#     lambda too low, or the features were already domain-invariant, which is
#     itself the publishable finding and is worth confirming with the
#     measurement script rather than more training.
# Sweep plan that fits in one GPU slot each: --uda_weight 0.1 first, then 0.5.
# Only once a weight looks alive should seeds 1 and 2 be spent.
#
# CORAL CONDITIONING, worth knowing before trusting ARM=coral. At batch 16 the
# sample covariance of 1536-dimensional pre-logits features has rank 15 out of
# 1536, so ||C_s - C_t||_F^2 is dominated by sampling noise rather than by any
# real difference between the domains, and the gradient mostly chases that
# noise. MMD with an RBF kernel does not have this problem (it needs no
# covariance estimate). If CORAL is run at all, --uda_taps stage0,stage1 (192
# and 384 channels) is far better conditioned than prelogits.
#
# MEMORY. The target batch is a second forward pass, so peak memory is roughly
# 1.6-2x the baseline's. The log prints peak_mem_gib at the end of every epoch.
# If epoch 0 dies with CUDA OOM, set --uda_target_batch 8 (statistics get
# noisier but the estimator stays unbiased in the mean) or drop --batch_size to
# 12 with --accum_iter 5 (which also changes the baseline arm and therefore
# needs its own control).

set -euo pipefail
: "${ARM:=mmd}"
: "${SEED:=0}"
: "${LAMBDA:=0.1}"
: "${DRY:=0}"      # DRY=1 validates the command line and data layout on a CPU
                   # node (no GPU, no model, no image decoding) and exits:
                   #   srun -p qCPU24 --mem=8G -t 5 bash -c \
                   #     'ARM=mmd DRY=1 bash scripts/95_uda_align.slurm'

PY=/home/users/sthummala2/.conda/envs/retfound/bin/python
cd /home/users/sthummala2/brset-convnextv2

# Everything below this line is byte-identical to the budget-0 label-efficiency
# baseline (scripts/71, BUDGET=0) except for the UDA_FLAGS block.
BASE=(
  --model convnextv2_large.fcmae_ft_in22k_in1k_384
  --data_path /home/users/sthummala2/brset-convnextv2/data/finetune_joint
  --nb_classes 2 --input_size 512 --resize_size 560
  --batch_size 16 --accum_iter 4 --epochs 25 --warmup_epochs 3
  --lr 3e-5 --weight_decay 0.1 --drop_path 0.3 --mixup_alpha 0.2
  --label_smoothing 0.1 --ema_decay 0.999 --tta flip4 --thr_bootstrap 200
  --ckpt_freq 0 --num_workers 8 --amp_dtype bf16
  --loss focal --focal_gamma 2.0
  --target_budget 0 --budget_seed 0
  --seed "${SEED}"
  --output_dir /home/users/sthummala2/brset-convnextv2/results
)
# The sampler block is separate only so the 'prior' arm can swap it without
# leaving an empty argument in the array.
SAMPLER=( --no_oversample )
TARGET=/home/users/sthummala2/brset-convnextv2/data/finetune_mbrset_multilabel/train

case "${ARM}" in
  mmd)    # PRIMARY. Multi-scale RBF MMD on pre-logits features.
    UDA_FLAGS=( --uda_loss mmd --uda_weight "${LAMBDA:-0.1}" --uda_taps prelogits
                --uda_scale_mode relative --uda_target_path "${TARGET}"
                --uda_target_batch 16 --uda_rampup_epochs 3 --uda_num_workers 4 )
    TASK="uda_mmd_w${LAMBDA:-0.1}_s${SEED}" ;;
  coral)  # Deep CORAL (covariance only), tapped at stage0+stage1 where a
          # 16-sample covariance is not hopelessly rank-deficient. Use
          # --uda_taps prelogits for the literally-published placement.
    UDA_FLAGS=( --uda_loss coral --uda_weight "${LAMBDA:-0.1}"
                --uda_taps stage0,stage1 --uda_scale_mode relative
                --uda_target_path "${TARGET}" --uda_target_batch 16
                --uda_rampup_epochs 3 --uda_num_workers 4 )
    TASK="uda_coral_w${LAMBDA:-0.1}_s${SEED}" ;;
  stage)  # MMD on the last two ConvNeXt stages as well, since the measured
          # domain gap is low-level (noise and contrast), not semantic.
    UDA_FLAGS=( --uda_loss mmd --uda_weight "${LAMBDA:-0.1}"
                --uda_taps stage2,stage3,prelogits --uda_scale_mode relative
                --uda_target_path "${TARGET}" --uda_target_batch 16
                --uda_rampup_epochs 3 --uda_num_workers 4 )
    TASK="uda_mmd_stages_s${SEED}" ;;
  prior)  # MMD plus source resampling to the target's 23.4% DR rate. NOTE this
          # drops --no_oversample (apply_pos_rate is ignored when the sampler is
          # off, see scripts/30) and uses one scalar derived from target labels,
          # so it needs its own control: the same command with
          # --uda_loss none --uda_weight 0.
    UDA_FLAGS=( --uda_loss mmd --uda_weight "${LAMBDA:-0.1}" --uda_taps prelogits
                --uda_scale_mode relative --uda_target_path "${TARGET}"
                --uda_target_batch 16 --uda_rampup_epochs 3 --uda_num_workers 4 )
    SAMPLER=( --target_pos_rate 0.234 )
    TASK="uda_mmd_prior_s${SEED}" ;;
  st)     # SECONDARY. Confidence-thresholded self-training on top of MMD.
          # Confirmation bias is the obvious failure mode: at a 0.9 threshold a
          # model that misses 42 of 159 positives will mostly pseudo-label
          # negatives and reinforce exactly that. The log prints
          # st_confident_fraction and st_pseudo_positive_rate every epoch;
          # compare the latter against mBRSET's 23.4% prevalence.
    UDA_FLAGS=( --uda_loss mmd --uda_weight "${LAMBDA:-0.1}" --uda_taps prelogits
                --uda_scale_mode relative --uda_target_path "${TARGET}"
                --uda_target_batch 16 --uda_rampup_epochs 3 --uda_num_workers 4
                --st_weight 0.5 --st_conf 0.9 --st_start_epoch 8
                --st_rampup_epochs 5 --st_teacher ema )
    TASK="uda_mmd_st_s${SEED}" ;;
  off)    # CONTROL. Reproduces results/lebase_b0_s0 through this entry point;
          # the flags are off so the baseline epoch loop runs unchanged.
    UDA_FLAGS=()
    TASK="uda_off_control_s${SEED}" ;;
  *) echo "unknown ARM=${ARM}" >&2; exit 2 ;;
esac

echo "=== ${TASK}: arm=${ARM} seed=${SEED} ==="
echo "must beat DR AUC 0.9246 (0.9104 + one noise floor) to mean anything"
DRY_FLAG=()
[ "${DRY}" = "1" ] && DRY_FLAG=( --uda_parse_only )

"$PY" scripts/94_uda_align.py "${BASE[@]}" "${SAMPLER[@]}" "${UDA_FLAGS[@]}" \
  --task "${TASK}" "${DRY_FLAG[@]}"
