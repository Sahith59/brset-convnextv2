# Steps 0–4 explained in simple language

## Research question

BRSET images and mBRSET images are photographs of the same kind of object—the retina—but they were collected with different devices and under different acquisition conditions. We want a model that detects image-level diabetic retinopathy (DR) and macular edema (ME) reliably on mBRSET. The current protocol allows labeled mBRSET training data, so this is supervised cross-device transfer rather than a true zero-shot experiment.

## Step 0 — understand and show the degradation

**Question:** Can we alter a BRSET image so that simple aspects of its appearance move toward mBRSET?

The existing transform changes blur, uneven lighting, bright spots, dark holes, halo, brightness, contrast, saturation and sensor noise. We checked real dataset files, regenerated original/transformed examples, documented every setting and sent Dong a three-page report. On patient-disjoint training images, the frozen transform reduced the five-statistic distance from 6.742 to 1.261 under the old resize and from 8.474 to 4.294 under the classifier-aligned resize/crop.

**Why it matters:** It establishes that the code changes the intended measurable appearance properties and gives Dong concrete images to inspect.

**Result:** Positive for descriptive appearance matching. It did not test classification, lesion preservation or novelty.

**Conclusion:** The degradation idea was plausible enough to test under a fair classifier protocol, but it was not yet a solution.

## Step 1 — define a fair experiment

**Question:** How do we stop changes in splits, preprocessing, training time or metric calculation from creating misleading comparisons?

We froze the patient-level fit/validation/test roles, labels, preprocessing, ConvNeXt V2 initialization, optimizer, focal loss, Mixup, update count, EMA selection, threshold selection, metrics, random seeds and artifact hashes. Selection means choosing the checkpoint and thresholds using mBRSET validation. Assessment means measuring the frozen choice on the mBRSET test set.

**Why it matters:** A method comparison is meaningful only when the method is the intended difference.

**Result:** Passed as an experimental-design and engineering gate. It is not an accuracy result.

**Conclusion:** Later results can be audited and compared under a common contract.

## Step 2 — establish strong ordinary baselines

**Question:** Before proposing a new method, how well do three standard training strategies work?

All three used the full original training pools and were repeated at seeds 0, 1 and 2:

| Baseline | Training strategy | Mean DR F1 | Mean ME F1 |
|---|---|---:|---:|
| B0 | 3,402 labeled mBRSET images only | 0.8105 | 0.8240 |
| B1 | 11,372 BRSET + 3,402 labeled mBRSET jointly | **0.8338** | **0.8619** |
| B2 | 11,372 BRSET first, then 3,402 mBRSET | 0.8139 | 0.8184 |

B1 exceeded B0 for both labels in all three seeds. Mean B1−B0 F1 was +0.0233 for DR and +0.0379 for ME.

**Why it matters:** Any new method must improve on B1 rather than on a weak or incomplete baseline.

**Result:** Positive for joint training. This result is a strong baseline finding, not a novel method.

**Conclusion:** Natural joint training B1 became the reference for later experiments.

## Step 3 — test whether balancing explains the joint-training gain

**Question:** Is B1 better simply because of how often it sees each domain or each DR/ME label combination?

We compared natural B1 sampling with equal-domain sampling, target-label-composition sampling and their combination while keeping the number of updates fixed. Only equal-domain C1 triggered replication after seed 0. Across three seeds, C1−B1 validation F1 was +0.0035 for DR and +0.0079 for ME. DR changed direction across seeds, and mean AUROC was lower for both labels. All B1/C1 training curves peaked before the final quarter and declined afterward.

**Why it matters:** It prevents us from attributing a sampling effect to a degradation or representation method.

**Result:** Mixed/negative as a proposed improvement. Equal-domain sampling did not provide a clear, stable advantage. Positive as a scientific control because it ruled out a tempting explanation.

**Conclusion:** Retain the simpler natural B1 sampler. Do not spend compute on the planned double-length exposure run.

## Step 4 — test target-calibrated source augmentation

**Question:** When everything from B1 is fixed, does changing only BRSET training views improve real mBRSET DR/ME classification?

Before classifier training, we checked whether unequal DR/ME prevalence was driving the measured appearance difference. Reweighting both datasets to the same joint DR/ME composition changed every appearance gap by less than 0.05 mBRSET standard deviations and reversed none. We therefore used one global target-training-only refit.

The new refit used statistics from all 3,402 mBRSET training images, 96 BRSET patients for fitting and 96 different BRSET patients for verification. Held-out appearance distance was 6.0661 without transformation, 0.9018 for the full fitted transform and 0.9421 without overlays. The overlay-free result shows that strong appearance matching does not require artificial spots, holes or halo under this narrow objective.

The classifier screen compares:

- A1: a release-range FundusAug artifact component;
- A2: the full target-fitted transform; and
- A3: the target-fitted transform without synthetic overlays.

All arms use the same 14,774-image B1 pool, 5,775 updates, seed-0 initialization and 725-image mBRSET validation set. The mBRSET test set remains unused.

**Why it matters:** This is the first direct test of whether the degradation work helps diagnosis rather than merely making global image summaries look closer.

**Result:** Positive for held-out appearance matching, but negative as a diagnostic-improvement screen. Compared with B1 on mBRSET validation, A1/A2/A3 changed DR F1 by +0.0069/+0.0036/+0.0062 and ME F1 by -0.0028/-0.0004/-0.0004. None reached the frozen +0.01 F1 replication trigger. A3 increased DR AUROC by +0.0038, but that single-seed signal is too small to claim an improvement.

**Conclusion:** Do not automatically replicate or test-assess these arms. Matching five global appearance statistics did not produce a large diagnostic gain. Preserve the negative evidence and move to a separately frozen content-preserving mechanism.

## Overall position

We have not lost the research progress. Steps 0–3 established and audited the appearance premise, experimental contract, strongest baseline and sampling explanation. Step 4 showed that strong global appearance matching alone is insufficient for a large diagnostic improvement. The work has produced useful positive and negative evidence, but a novel contribution has not yet been demonstrated.
