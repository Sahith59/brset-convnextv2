# Step 5 plan draft: content-aware latent transition

**Status:** Revised after the allocated appearance-error gate. CPU evidence gate complete; no Step-5 GPU work has launched.

## Plain-language goal

The Step-4 simulator changed how BRSET images looked but did not reliably improve DR and ME decisions. A new validation-only audit also found that the five global appearance measurements did not reliably explain B1 errors. Step 5 therefore moves away from degradation as its first mechanism and asks whether global pooling is losing small, disease-specific evidence and whether a small known-device residual can use source data without forcing both cameras through identical features.

## Work packages

### 5A. Repair the paper baseline

- Add a strict source-only run: BRSET train for fitting, BRSET validation for checkpoint/threshold selection, mBRSET test only after choices are frozen.
- Keep it distinct from B1, which uses labeled mBRSET training images.
- Match initialization, optimizer, update count, preprocessing and evaluation wherever the different selection domain permits.

### 5B. Low-cost top-conference comparators and diagnostics

- Extract frozen EMA pooled features and logits for B1 on fitting and selection roles on an allocated GPU node.
- Verify that the unmasked cached features reproduce the original classifier logits/probabilities within a frozen tolerance.
- Reproduce CVPR 2026 GFP: group final features, rank groups by fitting-set `(AUROC + AUPRC)/2`, choose group size/minimum keep ratio on mBRSET validation, and report DR and ME separately.
- Do not assess test or claim novelty. GFP is a published comparator and redundancy diagnostic.
- From the same frozen backbone, compare ordinary global pooled features with label-specific spatial top-k evidence heads using patient-grouped fitting/validation roles.
- Do not proceed to a patch mechanism unless the spatial diagnostic shows a prespecified validation signal.

### 5C. Conditional mechanism screen

- P1: label-specific patch evidence head, shared across BRSET and mBRSET.
- P2: known-device residual adapters with ordinary global pooling.
- P3: label-specific patch evidence plus known-device residual adapters.
- Preserve the natural B1 supervised fitting pool, validation set, EMA evaluation and update budget.
- Use the known acquisition-domain label at training and inference; do not learn a router.
- Run at most three nonexclusive single-A40 jobs concurrently after CPU and GPU smoke gates pass, and only if the frozen spatial diagnostic passes.

### 5D. Decision gate

Replicate only an arm with one F1 delta at least `+0.01`, the other at least `-0.01`, and both AUROC deltas at least `-0.01` versus B1. P3 must also beat P1 and P2 for a patch/adaptation interaction to remain defensible. Seeds 1 and 2 and test assessment are conditional.

## Evidence boundaries

- The primary project remains supervised cross-device joint training until Dong confirms a zero-target-label question.
- Patch attention is not lesion localization proof without lesion annotations or expert review.
- Generic patch attention, shared/private encoders and domain adapters are established prior art.
- The working mechanism is not an established novelty claim.
- The existing mBRSET test is historically reused and must be disclosed.

## Immediate implementation checklist

1. Freeze exact BRSET source validation counts and patient separation.
2. Freeze pooled and spatial feature extraction and prove cached unmodified logits reproduce B1.
3. Freeze GFP grouping/selection and label-specific spatial-head training/selection without test access.
4. Use the diagnostic result to decide whether P1/P2/P3 are justified.
5. If justified, freeze adapter placement, patch count, loss and parameter budget before seed-0 results.
6. Add resume equivalence, no-test-access and exposure-ledger tests, then measure runtime on one A40.

## Superseded branch

W1/W2/W3 action-conditioned degradation is suspended by `appearance_error_gate.json` and `APPEARANCE_ERROR_GATE_REVIEW.md`. It must not be launched as the first Step-5 screen. Reconsider it only if later evidence identifies a localized acquisition-transition problem.
