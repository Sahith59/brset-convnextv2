# Frozen GFP comparator protocol

**Frozen before GFP computation:** September 15, 2026

Use the private B1 seed-0 pooled features produced under `feature_v1.json`. The backbone and original two-output classifier head remain frozen. The fitting cohort contains 14,774 BRSET/mBRSET training images; model/configuration selection uses only the 725-image mBRSET validation split; no assessment/test feature exists.

For each contiguous group size in `{1, 4, 16, 64, 128}`, rank groups separately for DR and ME using their solo fitting-set mean `(AUROC + average precision)/2` under the original classifier weight and bias. Evaluate every ranked prefix once. For each minimum keep ratio in `{0, .25, .50, .75}`, select the fitting-best eligible prefix; exact ties retain more features. Apply the two label-specific masks to four-view validation features.

Select one common group-size/minimum-ratio configuration by mean DR/ME validation AUROC. Exact ties retain more total label features, then use the smaller group size. Choose DR/ME thresholds on validation only after the configuration is fixed. Compare with unmasked B1 validation predictions and confirm that the cached pooled features reproduce the saved B1 probabilities.

Offline group masking reapplies the saved head in float32 to pooled values emitted under bf16 inference. The maximum probability difference from the saved end-to-end bf16 B1 prediction must be at most `0.01`; report the observed value. The matched unmasked float32 head receives its own validation-selected thresholds before comparison with GFP.

Call the comparator screen positive only if one validation F1 difference is at least `+0.01`, the other is at least `-0.01`, and both AUROC differences are at least `-0.01`. This is an engineering screen on reused development data, not statistical significance or a proposed contribution.
