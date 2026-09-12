**Data usage clarification and next comparison decision — September 10, 2026**

The user's concern is valid. Current B0/B1/B2 are reduced-training-pool development experiments; they are not full-data baselines. The additional split was a conservative design choice by Codex, not a requirement to withhold 40% of source training images indefinitely.

**Where the images went**

BRSET raw 16,266 → eligible multilabel cohort 16,258 (eight historical exclusions). Original patient-level roughly 70/15/15 split: train 11,372, validation 2,451, test 2,435.
mBRSET raw 5,164 → quality=yes 4,872 → eligible labeled cohort 4,859. Original patient-level roughly 70/15/15 split: train 3,402, validation 725, test 732. The source and target quality inclusion rules differ; current outcomes do not cover all ungradable handheld images. Counts/filtering were documented in the prior audit and preparation scripts 06/13.

Then only the original TRAIN pools were subdivided by patient, approximately 60/20/20:

| Dataset | Fit | Selection | Assessment | Sum |
|---|---:|---:|---:|---:|
| BRSET | 6,826 | 2,261 | 2,285 | 11,372 |
| mBRSET | 2,043 | 686 | 673 | 3,402 |

Thus the current fitting portion is approximately 0.70×0.60=42% of the eligible dataset. The original validation/test sets still exist and remain excluded from these runs. BRSET selection/assessment roles are not used to select or score the primary target baselines. Current model selection and primary assessment use mBRSET only.

The reason for the extra target partition was to separate fitting from checkpoint/threshold choice and assessment while keeping historically reused validation/test sets out of routine new model selection. This creates useful development boundaries, not a pristine independent study. Historical use of data still needs disclosure. Holding out source patients also restricts available training data and is not necessary merely to evaluate on mBRSET. The existing reduced-pool protocol is valid for its stated comparison, but should not become the only evidence for a best-use-of-data claim.

**Current experiments**

B0: fit on 2,043 mBRSET images/539 patients. B1: fit on 6,826 BRSET + 2,043 mBRSET images (8,869 images, 4,118 dataset-specific patient IDs). B2: source phase on 6,826 BRSET images/3,579 patients, then target phase on 2,043 mBRSET images/539 patients. All three select on 686 mBRSET images/180 patients and assess on 673 mBRSET images/180 patients. B2 source stage uses a fixed update budget and final source weights, not BRSET validation selection.

**Decision before further expensive replication**

Finish the three currently running fixed-protocol seed-0 runs; do not change data midway or reinterpret their assessment as full-data performance. Before committing to seeds 1/2 and augmentation comparisons, explicitly choose the training-data budget for that comparison series. Include a matched full-original-training-pool comparison in the paper plan: target-only 3,402 images; joint 11,372+3,402=14,774; sequential source 11,372 then target 3,402. Baseline and proposed method must share that larger training pool, selection rules, evaluation patients and update budget within their comparison. This is a planned comparison, not submitted jobs or a locked revised protocol.

If returning to original 725-image validation/732-image test for that series, disclose their historical reuse and treat results accordingly; calling them untouched would be wrong. Seek genuinely independent confirmation where feasible. Do not compare full-pool scores on the old 732-image test directly with reduced-pool scores on the new 673-image assessment as if the difference isolates training size. A same-target-development-split full-source control is another possible resource ablation; select the exact comparison before its outcomes, rather than automatically adding every possible run.

More source/target training data may help, but no guaranteed increase in F1 follows: cohort differences, labels, sampling and optimization matter. More data does not justify training on the same images used to score that model.

**Degradation timing**

Preparation can proceed while GPU baselines run: audit exact augmentation integration, decide the common data budget, specify balance/exposure controls and a FIT-only fitting/appearance-check manifest, and arrange independent diagnostic-preservation review. The next GPU augmentation comparison follows baseline sanity checks and a frozen common comparison protocol. It need not wait for every baseline seed if that screening design is explicit, but must not mix new data budgets with old baselines and call the result an augmentation effect. Current three-GPU cap remains; do not launch a fourth concurrent GPU job.

Existing PDF appearance validation remains separate and unchanged. Its old fitted parameters must not be used in a split where fitting included evaluation patients. No promise that synthetic spots preserve disease information is made. The first augmentation classification experiment can follow these gates, potentially the next day if jobs/checks proceed normally; it is not yet scheduled or guaranteed.

**Status snapshot**

At 2026-09-10 15:17 UTC: B0 last logged 450/5775 updates; B1 200/5775; B2 200/5775 in source phase. All three running; checked error logs empty and sampled losses finite. Throughput ~4.33–4.35 seconds/update. Rough training finish 22:00–23:00 UTC September 10; assessment job 4372454 follows all successes, subject to queue/runtime variability. No diagnostic gain is established by these early logs. See progress_20260910_1516.json.
