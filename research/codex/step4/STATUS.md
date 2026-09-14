# Step-4 status

## Design and confounding audit completed — September 14, 2026

Step 3 is closed and natural joint B1 is the reference. No Step-4 classifier training or test assessment has started.

The closest-work review was extended through September 14 primary sources. It confirms that generic fundus artifact augmentation, consistency, grade-conditioned diffusion generation, retinal structure-preserving translation, clinical preservation benchmarking and saliency-preserving augmentation are occupied ideas. A defensible method must be more specific and empirically isolated.

The first Step-4 task was a training-only appearance/label-composition audit. It tested whether the global BRSET-to-mBRSET appearance gap changes after holding the joint DR/ME composition constant. This was necessary because the current five-statistic degradation fit might otherwise mix acquisition appearance with disease-prevalence differences.

Allocated CPU job `4393429` completed exit `0:0` in 5:04 on `arctrdcn001`; stderr is empty. It measured all 11,372 BRSET and 3,402 mBRSET original-training images under classifier evaluation geometry. Manifest/source hashes, stratum totals, output privacy and arithmetic checks pass.

Standardizing both domains to their common pooled DR/ME composition changed the five target-minus-source gaps by only 0.0022–0.0438 mBRSET standard deviations and reversed none. This supports a global training-only refit; no label-standardized classifier arm is needed. This is descriptive evidence, not a causal camera/population decomposition.

Draft protocol: `STEP_4_PLAN_DRAFT.md`. Audit review: `APPEARANCE_LABEL_AUDIT_REVIEW.md`. Next freeze the release-range FundusAug artifact comparator, global fitted transform, overlay-free fitted control, preservation measurements and exact transformation contract before classifier GPU launch.

## Implementation gate active — September 14, 2026

The advisor transcript is now mapped to completed, partial and open deliverables in `meeting/DONG_TRANSCRIPT_AUDIT_2026-09.md`. The top-conference literature audit identifies CheXWorld (CVPR 2025) as the most plausible world-model reference and keeps GenDeg (CVPR 2025), the ICCV 2025 universal degradation model and DuDoNet (CVPR 2019) as separate concepts. Full world/diffusion-model development is conditional on Step-4 evidence.

Allocated CPU refit job `4394917` completed exit `0:0` in 36:46. On 96 held-out BRSET training patients, the five-statistic distance was 6.0661 unchanged, 4.2108 with historical parameters, 0.9018 with the new full fit and 0.9421 with the new overlay-free fit. This is appearance-only evidence. The private qualitative gallery showed no implementation failure, but full/A1 overlays can add or obscure lesion-like structures; the overlay-free control remains necessary.

The initial smoke `4395006` exposed that the inherited smoke helper mapped the new A-arm names to target-only data (`fit_images=3402`). It was cancelled and archived as invalid before completing an arm. The full-run helper would have selected both domains, but the failed smoke did not test source transformations and cannot serve as a gate. The corrected code forces every Step-4 arm to use B1 joint phases. Corrected CPU preflight `4397263` completed exit `0:0`, verifies all arms use the 14,774-image joint pool in both smoke and full modes, and reconfirms target-pipeline bitwise equality. Corrected one-A40 smoke `4397264` completed exit `0:0` in 10:17. Cross-arm verification passed: identical 292-source/92-target draw exposure, finite optimization, 725-image validation inference and no assessment for all three arms.

The A1 label is narrowed to an independently implemented, release-range-faithful **FundusAug artifact component**, not full GDRNet. Common B1 geometry, color jitter, optimizer, loss, sampler and update count remain fixed. The source-only operation is applied after the random 512 crop and before the common flip, rotation and color jitter.

Seed-0 validation wave `4397838` is running for A1/A2/A3 on three nonexclusive nodes, one A40 per arm. At the September 14 19:53 UTC audit, every arm had reached update 2,400/5,775 (41.6%); logged losses were finite and all three stderr files were empty. The most recent selection checkpoint was update 2,310. Its threshold-0.5 F1 values are training-health observations only because the frozen finalizer has not selected per-label thresholds or compared the arms with B1. Validation-only summary `4397839` depends on all three tasks succeeding. No Step-4 test assessment is submitted. At observed throughput, completion is expected around 00:00–01:30 UTC September 15, subject to filesystem and node conditions.

Dong explicitly requested generated/degraded examples and qualitative validation at transcript times 14:04–15:57, then returned to validation and generative alternatives at 22:34–23:04. The previously sent Step-0 PDF contained verified examples. Advisor-only DOCX/PPTX companions with four current Step-4 rows were generated by Slurm CPU job `4399664`; structural/hash checks pass. They remain private and Git-ignored because they embed licensed training images. Refresh them with the verified seed-0 result before the next advisor update.
