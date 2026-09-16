# Step 5 diagnostic review

**Evidence date:** September 15, 2026
**Decision:** close GFP and simple top-k spatial pooling as negative mechanism screens; do not launch P1/P2/P3.
**Current direction (September 16):** The historical RETFound plan later in this review is superseded by the advisor correction at the end. The active backbone is ConvNeXt V2 Large.

## What was tested

Step 5 asked whether the remaining BRSET/mBRSET problem could be explained by either redundant final channels or loss of small local evidence during global averaging. The exact B1 seed-0 EMA checkpoint was frozen. Its features were extracted for 14,774 fitting images and 725 mBRSET validation images; no assessment feature was created. The cached four-view logits and probabilities reproduced the saved B1 values with maximum absolute error 0.0.

The separate strict source-only baseline trained and selected only on BRSET, then received one locked assessment on the historical 732-image mBRSET test split. This baseline is supporting gap evidence; it is not a target-labeled competitor or the proposed method.

## Finding 1 — strict source-only transfer misses positive cases

| Label | F1 | AUROC | Average precision | Precision | Sensitivity | Specificity | FN / positive |
|---|---:|---:|---:|---:|---:|---:|---:|
| DR | 0.7218 | 0.9100 | 0.8353 | 0.8972 | 0.6038 | 0.9808 | 63 / 159 |
| ME | 0.6972 | 0.9734 | 0.7988 | 0.8261 | 0.6032 | 0.9880 | 25 / 63 |

The high precision and specificity alongside approximately 0.60 sensitivity show a conservative operating point transferred from BRSET. AUROC remains much higher than F1, especially for ME, so the model retains ranking information but its source-selected decision boundary and representation do not recover enough target positives. This is a source-to-target dataset gap that may include device, site, population, prevalence and unmeasured factors; it is not a camera-only causal estimate.

For descriptive context only, target-labeled joint B1 corrected 34 source-only DR false negatives while introducing two new DR false negatives; it corrected 15 ME false negatives and introduced none. Because B1 uses labeled mBRSET fitting data, this post-hoc historical-assessment comparison cannot select the next method or be presented as an equal-resource contest.

## Finding 2 — GFP does not identify removable harmful channels

The independently implemented CVPR 2026 GFP comparator selected 16-channel groups. It retained 1,520 of 1,536 DR features and all 1,536 ME features. Relative to the matched unmasked frozen head:

| Label | ΔF1 | ΔAUROC | ΔAverage precision |
|---|---:|---:|---:|
| DR | 0.0000 | +0.00008 | +0.00019 |
| ME | 0.0000 | 0.00000 | 0.00000 |

The frozen screen failed. The result does not show that feature pruning is universally ineffective; it shows that the final B1 channels, original head and specified GFP search do not contain a useful pruning gain on this development split. Nearly every feature was retained, so simple final-channel redundancy is not a supported explanation for the remaining errors.

## Finding 3 — top-1 local evidence changes ranking but not the decision metric

Three deterministic small-head seeds compared a matched global-average head (G0) with label-specific top-k spatial heads. Mean DR/ME AUROC selected `k=1` before applying the frozen gate.

| Label and metric | G0 mean | Top-1 mean | Difference | Patient-bootstrap 95% interval for difference |
|---|---:|---:|---:|---:|
| DR F1 | 0.8647 | 0.8635 | −0.0012 | [−0.0205, +0.0186] |
| DR AUROC | 0.9530 | 0.9631 | +0.0101 | [+0.0011, +0.0220] |
| DR average precision | 0.9281 | 0.9348 | +0.0067 | [−0.0038, +0.0191] |
| ME F1 | 0.8431 | 0.8480 | +0.0048 | [−0.0197, +0.0314] |
| ME AUROC | 0.9903 | 0.9867 | −0.0036 | [−0.0082, −0.00004] |
| ME average precision | 0.9214 | 0.9012 | −0.0202 | [−0.0512, +0.0033] |

DR ranking improved consistently across all three head seeds, and DR sensitivity rose by 0.0152 on average, but F1 did not improve because additional false positives reduced specificity. ME showed no stable F1 gain and its AUROC decreased consistently. The selected patch is therefore a ranking cue, not a reliable two-label decision mechanism.

The top-1 result must not be called lesion localization: no lesion annotations were used, and a high-scoring patch can reflect optic disc, illumination, camera boundary, pathology or another correlated structure.

## Frozen decision

The spatial gate required at least +0.01 F1 for one label, at least −0.01 F1 for the other and no worse than −0.01 AUROC for either. Mean DR/ME F1 changes were −0.0012/+0.0048, so the gate failed. GFP also failed its matched screen. Under the predeclared plan:

- do not launch P1 patch-evidence training;
- do not launch P2 device adapters as part of the P1/P2/P3 interaction study;
- do not launch P3 combined training;
- do not access the test split for either diagnostic;
- preserve DR top-1 AUROC improvement as a secondary mechanistic observation, not a method claim.

## What the accumulated failures rule out

The evidence now argues against four simple explanations: unequal sampling alone, target-like global appearance alone, removable final-channel redundancy, and global pooling alone. None of these statements proves that the corresponding factor never matters. Each is restricted to the tested implementation, cohorts and metrics.

The strongest repeatable result remains ordinary supervised joint B1. The next planned branch is a protocol-matched published retinal foundation-model/backbone comparator, beginning with an audit of the existing companion RETFound project and available weights. No GPU training should begin until its split, label, initialization, input-resolution, update-budget and assessment boundaries are aligned with B1. A backbone comparison is a stronger baseline, not automatic novelty.

## Paper boundary

Current evidence supports a rigorous transfer study with a strong joint baseline, a quantified source-only gap and multiple falsified mechanisms. It does not yet support a novel-method performance claim. The advisor update should present this as disciplined narrowing of the problem and ask whether Dong prefers the remaining effort to prioritize a stronger retinal backbone comparison or a paper centered on the transfer analysis and negative controls.

## September 16 matched-comparator update

The required new implementation is now frozen and has passed its engineering gates. It uses the exact B1 joint fit cohort, any-DR and ME labels, the same 560-resize/512-crop input and 5,775-update budget, and validation-only development. Preflight job 4410136, full-backbone feasibility job 4410137 and exact-resume job 4410171 passed. The failure history and corrections are detailed in `RETFOUND_FEASIBILITY_REVIEW.md`.

Full seed-0 job 4411081 is running on one nonexclusive A40. Its first update is finite and its error log is empty. This is only implementation-health evidence; no RETFound performance or novelty claim is available until the completed validation output is independently verified.

## Advisor correction on September 16

The separate RETFound run completed after 5,775 updates, but the user clarified that Dong had previously set this backbone aside. Launching it was a direction error. Preserve the run record for provenance; exclude it from today's presentation, method selection, replication and assessment. The completed scientific Steps 2–5 used ConvNeXt V2 Large. The next branch returns to patient-level B1 error and clinical-evidence review under that backbone.
