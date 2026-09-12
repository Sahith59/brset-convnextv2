**Step 1: fixed development comparison protocol — September 10, 2026**

Status: experimental design and development manifests completed before new model runs. This is a development protocol, not a preregistration of an untouched external test. Training implementation and its smoke checks belong to Step 2. No GPU experiment was launched by Step 1. Amendments must be dated and made before examining the affected results.

**Question and scope**

Can an explicitly defined augmentation improve image-level any-DR and macular-edema prediction on real mBRSET images when BRSET labels and a fixed mBRSET training-label budget are available? Target-informed augmentation means its fitting may inspect only target FIT images. The overall study is supervised transfer, not zero-shot generalization. Source-trained models selected/calibrated on target labels must also be called target-selected, not zero-shot.

Keep the two existing binary manifest labels, including their original harmonization. BRSET standalone DR labels are not silently replaced with ICDR-derived labels; the prior audit documents disagreement. ME is the dataset image label, not an independently verified clinical/OCT endpoint. Predictions and confusion matrices are image-level. Uncertainty resamples patients, keeping all their images together. Namespace patient IDs by dataset; numerical IDs across datasets do not establish shared identity.

**Data boundaries**

`development_manifest_v1.csv` contains all 14,774 original training images, with five fixed patient folds and three roles. Seed 20260910. Stratify patients by their joint maximum DR/ME labels, falling back to DR strata only if a joint stratum has fewer than five patients; the actual rule and counts are saved in `preflight_v1.json`. This patient summary is used only to assign groups, not to replace image labels. Fold 0 is assessment; fold 1 is selection; folds 2–4 are fit. No seed/split search based on model performance.

| Domain | Role | Images | Patients | DR-positive images | ME-positive images |
|---|---|---:|---:|---:|---:|
| BRSET | Fit | 6,826 | 3,579 | 442 | 165 |
| BRSET | Selection | 2,261 | 1,193 | 147 | 51 |
| BRSET | Assessment | 2,285 | 1,194 | 159 | 58 |
| mBRSET | Fit | 2,043 | 539 | 481 | 180 |
| mBRSET | Selection | 686 | 180 | 170 | 62 |
| mBRSET | Assessment | 673 | 180 | 146 | 49 |

Fit images may update classifier weights, sampler weights, degradation parameters, teacher weights and any acceptance/rejection rule. Selection images may choose a checkpoint and decision thresholds; they never contribute training gradients or appearance-fit summaries. Assessment images score frozen checkpoints and thresholds after the predefined comparison batch is trained. No per-epoch assessment. Assessments used to choose later methods are development evidence, not unbiased final confirmation.

The old validation/test sets remain excluded from routine development. They have already influenced historical research choices and cannot be relabeled pristine. New development patients were also used in historical training; start new runs from external pretrained weights and report this history. A fresh external/appropriately independent final evaluation remains a paper-strengthening requirement, not something achieved by splitting old training images.

Old source task checkpoints and `fitted_degradation_params.json` are inadmissible as training initializations/augmentation fits for this protocol: their fitting may have included the new selection/assessment patients. Refit within FIT only. The frozen report result stays valid for its separately reconstructed exclusion design; it does not validate this new partition's future experiment.

**Initialization, common recipe and budget**

Backbone: `convnextv2_large.fcmae_ft_in22k_in1k_384`. Cached Hugging Face revision `4b4ce03d1a7a884ac2242b5fadcf7f0e55a66482`; independently hashed pretrained weights SHA256 `075703d09f806c09597fa3d68d8c6733a18ba1126adfbc1de97f8eb096402c2f`. Exact local path/environment versions are in `preflight_v1.json`. Replace its pretrained head with two outputs; save one initialized state per seed and reuse it across matched arms. Do not obtain a moving online `main` at launch.

Use bf16 on a supporting GPU; optimizer/model weights remain fp32. Microbatch 16, four accumulated microbatches per optimizer update: effective batch 64. Exactly 5,775 successful optimizer updates per single-stage arm, with 693 linear warmup updates from 0.1× learning rate to full rate, then cosine decay to zero over the remaining updates. This approximates the historical joint 25-epoch update budget (231 updates/epoch), rather than forcing differently sized data pools through equal epochs. New common runs consume 369,600 image draws. An epoch is one pass through a dataset; an update is one actual weight adjustment. Report both exposure and elapsed time.

AdamW: learning rate 3e-5, weight decay 0.1, betas (0.9,0.999), epsilon 1e-8; uniform weight decay across trainable parameters for continuity. Drop path 0.3; gradient norm cap 1.0. EMA decay 0.999, updated once after every successful optimizer update. EMA means a moving average of weights; evaluate only EMA to avoid unequal raw/EMA selection opportunities. No automatic mixed-precision fallback between arms. Stop on nonfinite loss/gradients and record the failure; do not silently skip difficult batches. Save full optimizer, scheduler, EMA, RNG and draw-counter state for exact continuation.

Training image preparation: RGB; bilinear square resize 560; uniform random 512 crop; horizontal flip probability 0.5; rotation uniform -15 to +15 degrees with nearest interpolation, no expansion and black fill; color jitter brightness/contrast 0.2 and saturation 0.1, no hue. Convert to tensor and normalize with mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. This retains the historical squash geometry as a controlled reference, not a claim it is optimal. Geometry changes require separate paired comparisons.

Mixup coefficient is sampled from Beta(0.2,0.2) within a microbatch, mixing images and labels with the same permutation. Then smooth targets: y' = 0.9 y_mix + 0.05. Retain the implementation's soft-target focal objective: b = BCEWithLogits(z,y'); L = mean[(1-exp(-b))^2 b], with no positive-class weight. For soft targets this is the historical BCE-modulated extension, not identical to the original hard-label focal derivation. All mechanism comparisons retain it unless a separately declared loss ablation is run.

Independent RNG streams: initialization, sample ordering, common transforms, Mixup and candidate transforms; derive them deterministically from training seed and named stream. Extra augmentation must not advance the sampler or Mixup streams. The same seeds do not guarantee bit-identical GPU training; log deterministic flags and hardware. Screening seed 0 first; every surviving primary comparison repeats seeds 1 and 2 before making a gain claim. Never select the best seed.

**Evaluation and thresholds**

Evaluation geometry: bilinear 560-square resize then center 512 crop, normalization above. Four-view flip averaging means arithmetic mean of sigmoid probabilities for original, horizontal, vertical and both flips. Use the same four views during checkpoint selection, threshold fitting and assessment. No augmentation or stochastic dropout during evaluation.

Evaluate every 231 successful updates, including update 5,775: 25 selection opportunities. Select the EMA checkpoint maximizing the mean of DR and ME AUROC on mBRSET SELECTION; exact ties choose the earlier update. This threshold-independent checkpoint rule changes the historical mixed selection rule and applies equally to all new matched arms. All outcomes on the selection set are optimization diagnostics, not held-out performance claims.

For the chosen checkpoint, choose each label's threshold on SELECTION only. Grid {0,0.001,...,1}; prediction is probability >= threshold. Maximize positive-class F1; ties within 1e-12 choose closest to 0.5, then the lower threshold. No bootstrap threshold smoothing in v1; it would add another procedure without making this small sample independent. Freeze thresholds before ASSESSMENT. Also report F1 at threshold 0.5, so threshold effects are visible.

Primary development outcome: DR positive-class F1 = 2TP/(2TP+FP+FN). Secondary: ME positive F1; per-label AUROC, average precision (AP, explicitly the chosen precision-recall summary), precision, sensitivity, specificity, class-macro F1 and confusion matrices. AUC measures ranking, not the quality of one threshold. Positive F1 differs from averaging positive and negative F1. Never collapse DR and ME into one improvement narrative. A clinical operating point remains unagreed; no sensitivity/specificity requirement is invented on Dong's behalf.

For paired model comparisons, resample the same ASSESSMENT patients with replacement for both models, retaining all images and patient multiplicity. Use 2,000 draws, RNG seed 20260911, fixed thresholds, and percentile 95% intervals for metric differences. Report undefined draws and their count for class-dependent metrics; never replace undefined AUC with zero. These intervals condition on the trained models and selected method; they do not capture all training/selection uncertainty. Report per-seed differences separately, not 3×images as independent observations.

**Predefined comparison order**

| Arm | Information/training | What this comparison establishes |
|---|---|---|
| B0 target-only | mBRSET FIT; external initialization; 5,775 updates | A target-supervised reference |
| B1 joint | Both FIT pools; external initialization; 5,775 updates | A joint-training strategy at matched total update budget |
| B2 source→target | BRSET FIT for 2,887 updates, then mBRSET FIT for 2,888; use final source state | Sequential transfer at matched total compute steps; target exposure differs from B0/B1 |

B2 starts from the same external initialization; source phase uses 3e-5 with 346 warmup updates, target phase uses 1e-5 with 347 warmup updates. Cosine decay within each phase. Reset optimizer/scheduler and initialize target EMA from the transferred final source raw weights. Evaluate/select only in the target phase at ceil(j×2888/25), j=1..25; 25 opportunities. B2 tests a prespecified training strategy, not the isolated causal effect of pretraining. Any added source pretraining outside the 5,775 updates must be reported as extra compute in a separate resource-unmatched arm.

B0/B1 sample by repeated independently shuffled passes through their allowed FIT image list, wrapping to complete all microbatches. No rare-class oversampling. B1 natural pool prevalence: report from the manifest, do not substitute equal-domain sampling. Step 3 specifies separate fixed-pool disease/domain balancing and source-repetition exposure controls before running them. Same epoch count, changed source/target ratio or extra source weights are not fair isolated augmentation comparisons.

Step 4 compares one frozen joint recipe under ordinary appearance augmentation, official FundusAug component and FIT-only refitted local augmentation. Exact additional transform probabilities and integration order must be pinned in a Step-4 supplement before these runs; these candidates are not launch-ready in Step 1. Match number of source views/draws and optimization updates. Adding extra views incurs compute and requires a matched-view control. Add a color/noise-only and simple severity-limited local control if proposing an artifact-selection mechanism. FundusAug alone is not full GDRNet. Consistency is a conditional subsequent ablation, not a bundled initial experiment.

**Decision rules, budget and completion boundary**

Initial screen: compare all three baseline strategies at seed 0; report all, then replicate the relevant reference and contender at seeds 1/2 before claiming gain. Engineering target for a later augmentation candidate: mean DR F1 improvement >=0.01 and positive DR difference in at least two of three seeds versus the declared reference; mean ME F1 loss no worse than 0.01. These are prioritization rules, not clinical noninferiority margins or significance tests. Compare against the strongest relevant ordinary/published-augmentation control before a contribution claim. If results are within these bands, retain the simpler method pending uncertainty/compute evidence.

Do not claim convergence from a best epoch alone. A bounded 11,550-update extension is a separate matched-budget experiment for both relevant arms if their final-quarter selection curves are still improving; declare that extension before reading its assessment. It uses a fresh full-length learning-rate schedule from the same initialization, not an undocumented checkpoint warm start. This addresses Dong's longer-training question without changing only one competitor's budget.

Historical joint log `logs/joint_4120451.out` shows roughly 40 seconds for 40 microbatches during the first epoch. A coarse planning estimate is ~6.4 GPU-hours for 23,100 microbatches, excluding selection inference, startup and I/O. Budget provisionally 8–12 GPU-hours per baseline run and 24–36 for the three seed-0 strategies; this is an estimate, not a measured runtime for the new trainer or all GPU types. A Step-2 allocated smoke/timing run must replace it before bulk submission. `sinfo` showed A40/L40S/A100-class resources, not a guarantee of immediate availability or account entitlement.

Step 1 completes the protocol, manifests, checks, pinned weight bytes and decision rules. Step 2 must implement a separate trainer that honors this contract: the old trainer always loads legacy test and must not be used unchanged. Before runs: verify decoded duplicate leakage (resolved paths alone are insufficient), pin actual initialized head states and code/config hashes, run CPU metric/split checks and an allocated GPU smoke/timing check, then prepare exact job commands and costs. No retrospective promise of novel accuracy is a launch prerequisite; a well-defined experiment is.

**Evidence files**

`preflight_v1.json`: manifest hashes, patient/class counts, cached weight hash, environment versions, original-code hashes and explicit incomplete checks. `development_manifest_v1.csv`: per-image assignment and resolved file paths. `development_counts_v1.csv`: human-readable denominators. `../tools/prepare_protocol.py`: deterministic reconstruction. Existing audit: `/home/users/sthummala2/research-audit-brset-20260909/AUDIT_REPORT.md`. No original result file was changed.
