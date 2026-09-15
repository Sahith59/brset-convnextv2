# Step 5 concrete execution plan

**Frozen planning date:** September 15, 2026  
**Paper target:** complete draft October 15; experimental/content freeze October 19; tentative ISBI deadline October 26.  
**Current state:** appearance-conditioned branch rejected; strict source-only design preflight passed; decoded-image integrity audit running. No Step-5 GPU job has launched.

## Paper question

The primary question remains supervised cross-device DR/ME classification with BRSET and labeled mBRSET training data. A separate strict source-only baseline quantifies how much performance is lost when no mBRSET training label is used. These settings must remain separate in the paper.

## Work and proportional budget

| Share of remaining experimental effort | Work | Concrete output | Stop/go decision |
|---:|---|---|---|
| 15% | Integrity and protocol gates | Source-only manifest, decoded duplicate audit, exact feature/checkpoint hashes, no-test contracts | No GPU work unless all checks pass |
| 20% | Strict source-only baseline | One seed-0 BRSET-fit/BRSET-validation model and one frozen mBRSET assessment | Publication baseline; replicate only if it becomes central to a source-only claim |
| 20% | Frozen B1 diagnostics | One feature-extraction pass; faithful GFP and global-versus-label-specific spatial evidence comparisons | No backbone retraining; decide whether a patch mechanism is justified |
| 30% | Conditional seed-0 mechanism screen | P1 patch evidence, P2 known-device adapter, P3 combined; maximum three nonexclusive one-GPU jobs | P3 must beat B1, P1 and P2 and pass the fixed F1/AUROC gate |
| 15% | Conditional replication and assessment | Seeds 1/2 for one qualifying method, then one gated test assessment and patient bootstrap | Stop the mechanism if it fails replication; do not search the test set |

The percentages are planning shares, not statistical weights. If the frozen spatial diagnostic fails, its 30% mechanism allocation moves to a stronger published retinal-backbone comparison and paper analysis rather than another speculative custom module.

## Phase 5A — strict source-only baseline

Exact verified cohorts:

- fitting: 11,372 BRSET training images from 5,966 patients;
- checkpoint and threshold selection: 2,451 BRSET validation images from 1,279 patients;
- assessment: 732 mBRSET test images from 193 patients;
- mBRSET images or labels used before assessment: zero.

Use the Step-2 initialization, ConvNeXtV2-Large, focal loss, ordinary augmentation, Mixup, AdamW, EMA, four-flip evaluation, 5,775 updates and seed 0. The only necessary protocol difference is that BRSET validation selects the checkpoint and thresholds. Report threshold-free AUROC/AP, F1 using BRSET-selected thresholds and F1 at 0.5. The test split is historically reused and must not be called pristine external validation.

## Phase 5B — frozen B1 diagnostics

Use the existing B1 seed-0 EMA checkpoint. Extract both:

- the final globally pooled feature vector used by the classifier; and
- the final spatial feature map before global pooling.

First prove that applying the stored classifier head to the cached unmasked pooled features reproduces the saved B1 logits/probabilities within a frozen numerical tolerance.

### GFP comparator

Partition the pooled feature vector into contiguous groups. On fitting data, rank groups by their solo mean DR/ME `(AUROC + AUPRC)/2`, then build the retained set greedily. Choose group size and minimum keep ratio using validation only. Report DR and ME separately. This is a CVPR 2026 method reproduction, not our contribution.

### Spatial-evidence diagnostic

Train lightweight label-specific heads on frozen spatial features and compare them with a matched linear head on frozen global features. Candidate retained-patch counts must be fixed to a small grid before results. Patient separation and validation-only model selection are mandatory. No test access.

Proceed to P1/P2/P3 only if one label F1 improves by at least `+0.01`, the other remains at least `-0.01`, and both AUROCs remain at least `-0.01` relative to the matched global frozen-feature head.

## Phase 5C — conditional mechanism

- **P1:** label-specific patch-evidence head shared across BRSET and mBRSET.
- **P2:** small known-device residual adapters with ordinary global pooling.
- **P3:** label-specific patch evidence plus known-device residual adapters.

The acquisition domain is known, so no learned router is needed. This avoids the historical training/inference routing mismatch. Match B1 initialization, draw budget, optimizer, validation protocol and inference rules. Report added parameters and runtime.

P3 becomes a replication candidate only if it passes the usual validation gate and exceeds both P1 and P2. That isolates whether the combination contributes beyond either established component.

## Claim boundary

A positive P3 result would establish an empirical interaction, not automatic novelty. Patch attention, lesion discovery, shared/private models and domain adapters all have prior art. The final method distinction must be written only after the diagnostic identifies what is new and another closest-work search confirms the wording.

No performance result is guaranteed. The plan maximizes the probability that every GPU run answers a paper-relevant question and prevents a weak result from being promoted through repeated tuning.

## Calendar

- **September 15–16:** integrity audit; freeze source-only trainer and feature-extraction contracts; CPU/GPU smoke checks.
- **September 16–17:** source-only seed 0 and B1 feature extraction, using at most two nonexclusive GPUs.
- **September 18:** GFP and spatial diagnostics; freeze the mechanism decision.
- **September 19–22:** conditional P1/P2/P3 seed-0 wave.
- **September 23–26:** conditional seeds 1/2.
- **September 27–30:** one frozen test assessment, patient bootstrap and error analysis.
- **October 1–12:** manuscript, figures, ablations and advisor revision.
- **October 15:** complete draft.
- **October 19:** hard experimental/content freeze.
- **October 26:** tentative ISBI submission deadline.

