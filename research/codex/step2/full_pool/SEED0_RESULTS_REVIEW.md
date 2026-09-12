# Full-pool baseline seed-0 review

Status: completed and independently recomputed on September 11, 2026. This is one training seed on a historically reused mBRSET test split, not a replicated or untouched external result.

| Arm | DR F1 | DR AUROC | DR AP | ME F1 | ME AUROC | ME AP |
|---|---:|---:|---:|---:|---:|---:|
| B0 target-only | 0.8000 | 0.9339 | 0.8831 | 0.8571 | 0.9687 | 0.9207 |
| B1 joint | 0.8312 | 0.9451 | 0.9037 | 0.9060 | 0.9945 | 0.9578 |
| B2 source then target | 0.8041 | 0.9410 | 0.8902 | 0.7805 | 0.9866 | 0.9070 |

Validation-selected thresholds were B0 DR/ME 0.305/0.639, B1 0.468/0.428 and B2 0.464/0.411. Selected checkpoint updates were B0 4389, B1 2772 and B2 3812. These differences are expected because selection chooses the best validation checkpoint under the common candidate schedule.

On 732 test images (159 DR positive; 63 ME positive), B0/B1/B2 DR false positives were 27/21/18 and false negatives 35/31/40. ME false positives were 1/1/12 and false negatives 15/10/15. These are image counts, not patient misses.

Observed B1-B0 changes: DR F1 +0.0312, DR AUROC +0.0112, ME F1 +0.0488 and ME AUROC +0.0258. Fixed-model patient-bootstrap 95% percentile intervals were DR F1 [+0.0013,+0.0660], DR AUROC [-0.0069,+0.0354], ME F1 [-0.0061,+0.1026] and ME AUROC [+0.0010,+0.0677]. These intervals condition on the three particular seed-0 models and do not include training-seed uncertainty.

Observed B2-B0 changes: DR F1 +0.0041 and ME F1 -0.0767. Observed B2-B1 changes: DR F1 -0.0271 and ME F1 -0.1255. Under this prescribed equal-update recipe, sequential fine-tuning did not improve seed-0 DR F1 materially over target-only and was worse for ME F1. This does not establish that sequential transfer is generally inferior; stage budget and learning-rate choices are part of the tested method.

Independent allocated-CPU job 4376489 recomputed all metrics from prediction arrays and verified prediction/checkpoint hashes, labels, cohort identity/order, 732 unique images, 193 patients and identical assessment data across arms. All original training/assessment jobs exited 0 and stderr files were empty. See `seed0_independent_verification.json` and `baseline_assessment_seed0.json`.

Interpretation: B1 is the strongest seed-0 arm across both F1 and AUROC and is the current promising baseline. No winner or paper claim is declared until seeds 1 and 2 are complete. The original mBRSET test was used historically, so call this a reused benchmark rather than untouched external confirmation. Do not use these test outcomes to tune Step-3 or Step-4 design.

Step-2 completion gate: seeds 1 and 2 for all three arms, gated assessments, independent provenance/metric checks, and a cross-seed summary. Step-3 design can be prepared while replications run, but method selection waits for validation-led evidence.
