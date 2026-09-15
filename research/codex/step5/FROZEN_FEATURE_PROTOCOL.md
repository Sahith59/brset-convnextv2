# Frozen B1 feature diagnostics protocol

**Draft frozen before extraction:** September 15, 2026  
**Scope:** B1 seed-0 checkpoint; BRSET/mBRSET fitting data and mBRSET validation only. No test access and no backbone update.

## Purpose

Answer two questions before training another full model:

1. Do redundant or harmful final channels limit B1, as proposed by CVPR 2026 GFP?
2. Does the final spatial map contain label-specific DR/ME evidence that global average pooling loses?

If neither diagnostic is positive, patch/adaptor model training is not justified.

## Extraction contract

- Load the exact B1 seed-0 EMA checkpoint and verify its hash against the saved selection metadata.
- Use the fixed 560 resize / 512 center crop and ImageNet normalization.
- Extract the final ConvNeXt spatial feature map before global pooling and the exact pre-logit pooled vector.
- For the 725-image mBRSET validation set, extract all four evaluation flips and verify that the stored B1 classifier head reproduces the saved per-view logits and averaged probabilities. Freeze maximum absolute tolerances after a small numerical smoke and before full extraction.
- Store features only in private shared storage. Public evidence contains cohort counts, shapes, hashes, reproduction errors, aggregate metrics and runtime.
- Fit data: 11,372 BRSET plus 3,402 mBRSET images. Selection data: 725 mBRSET images. Test data: none.

## GFP reproduction

The backbone and existing two-output classifier head remain frozen. Partition the pooled feature vector into contiguous groups with candidate group sizes `{1, 4, 16, 64, 128}` when divisible by the observed feature dimension. Candidate minimum keep ratios are `{0.00, 0.25, 0.50, 0.75}`.

For each label and group, zero all other groups and score the existing head on fitting data. Rank by the mean of AUROC and average precision. Starting from the minimum retained count, greedily add ranked groups and retain the prefix with the best fitting score. Use mBRSET validation only to select the common group-size/minimum-ratio configuration by mean DR/ME AUROC; ties choose more retained features, then smaller group size. Select DR/ME thresholds on validation after the mask is frozen.

Report GFP and unmasked B1 validation metrics for both labels. This is a published comparator and representation diagnostic, not a proposed contribution. No test evaluation occurs at this stage.

## Spatial-evidence diagnostic

Compare heads on the same frozen canonical spatial features:

- **G0:** matched trainable linear head on global-average pooled features;
- **G1(k):** separate DR and ME 1x1 patch scorers followed by mean of the top `k` patch logits, with `k` in `{1, 4, 8, 16, 32}`.

Use focal loss and fixed L2 regularization. Train the small heads on the natural BRSET+mBRSET fitting pool with three deterministic head seeds. Choose one common `k` on mBRSET validation by mean DR/ME AUROC; ties prefer larger `k`. Thresholds are selected independently per label on validation using the existing grid. The backbone remains frozen.

The spatial diagnostic passes only if the three-seed mean G1−G0 validation result has one F1 difference at least `+0.01`, the other at least `-0.01`, and both AUROC differences at least `-0.01`. This is an engineering screen. Attention or selected patches are not lesion-localization evidence.

## Decision

- GFP positive, spatial negative: investigate representation pruning or a stronger published backbone; do not build a patch mechanism.
- Spatial positive: freeze P1/P2/P3 full-training controls and repeat the closest-work audit.
- Both negative: stop custom feature mechanisms and frame the paper around rigorously measured transfer/baseline findings or replace the backbone with a published retinal foundation model if access and compute permit.

