# Step 5 plan draft: content-aware latent transition

**Status:** Direction selected from deep research; protocol and code are not yet frozen. No Step-5 computation has been launched.

## Plain-language goal

The Step-4 simulator changed how BRSET images looked but did not reliably improve DR and ME decisions. Step 5 will stop treating the changed image as merely another training example. Instead, the model will see the original image, its mobile-like version and the exact strength of the change. It will learn to recover the original image's feature representation while keeping the DR and ME decisions stable.

## Work packages

### 5A. Repair the paper baseline

- Add a strict source-only run: BRSET train for fitting, BRSET validation for checkpoint/threshold selection, mBRSET test only after choices are frozen.
- Keep it distinct from B1, which uses labeled mBRSET training images.
- Match initialization, optimizer, update count, preprocessing and evaluation wherever the different selection domain permits.

### 5B. Low-cost top-conference comparator

- Extract frozen EMA pooled features and logits for B1 on fitting and selection roles on an allocated GPU node.
- Verify that the unmasked cached features reproduce the original classifier logits/probabilities within a frozen tolerance.
- Reproduce CVPR 2026 GFP: group final features, rank groups by fitting-set `(AUROC + AUPRC)/2`, choose group size/minimum keep ratio on mBRSET validation, and report DR and ME separately.
- Do not assess test or claim novelty. GFP is a published comparator and redundancy diagnostic.

### 5C. Novelty-facing seed-0 screen

- W1: paired clean/degraded feature consistency, no action.
- W2: action-conditioned prediction of clean features from the degraded view.
- W3: W2 plus separate DR/ME prediction-preservation loss.
- Use the overlay-free Step-4 endpoint and a sampled identity-to-endpoint strength.
- Keep clean and degraded retinal geometry paired.
- Preserve the natural B1 supervised fitting pool, validation set, EMA evaluation and update budget.
- Run at most three nonexclusive single-A40 jobs concurrently after CPU and GPU smoke gates pass.

### 5D. Decision gate

Replicate only an arm with one F1 delta at least `+0.01`, the other at least `-0.01`, and both AUROC deltas at least `-0.01` versus B1. W3 must also beat W1 for the proposed action-plus-preservation mechanism to remain defensible. Seeds 1 and 2 and test assessment are conditional.

## Evidence boundaries

- The primary project remains supervised cross-device joint training until Dong confirms a zero-target-label question.
- The action parameters are target-calibrated from training images; they are not known physical camera parameters.
- Prediction consistency is not clinical proof of lesion preservation.
- The working method description is not an established novelty claim.
- The existing mBRSET test is historically reused and must be disclosed.

## Immediate implementation checklist

1. Freeze exact BRSET source validation counts and patient separation.
2. Define the action vector and identity-to-endpoint interpolation for every overlay-free operator.
3. Define which ConvNeXt feature tensor is predicted and the lightweight predictor size.
4. Fix loss weights before seed-0 results, using smoke-scale numerical checks rather than validation tuning.
5. Add resume equivalence, no-test-access and exposure-ledger tests.
6. Measure memory/runtime on one A40 before the three-arm launch.

