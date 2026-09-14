# Step-4 status

## Design and confounding audit completed — September 14, 2026

Step 3 is closed and natural joint B1 is the reference. No Step-4 classifier training or test assessment has started.

The closest-work review was extended through September 14 primary sources. It confirms that generic fundus artifact augmentation, consistency, grade-conditioned diffusion generation, retinal structure-preserving translation, clinical preservation benchmarking and saliency-preserving augmentation are occupied ideas. A defensible method must be more specific and empirically isolated.

The first Step-4 task was a training-only appearance/label-composition audit. It tested whether the global BRSET-to-mBRSET appearance gap changes after holding the joint DR/ME composition constant. This was necessary because the current five-statistic degradation fit might otherwise mix acquisition appearance with disease-prevalence differences.

Allocated CPU job `4393429` completed exit `0:0` in 5:04 on `arctrdcn001`; stderr is empty. It measured all 11,372 BRSET and 3,402 mBRSET original-training images under classifier evaluation geometry. Manifest/source hashes, stratum totals, output privacy and arithmetic checks pass.

Standardizing both domains to their common pooled DR/ME composition changed the five target-minus-source gaps by only 0.0022–0.0438 mBRSET standard deviations and reversed none. This supports a global training-only refit; no label-standardized classifier arm is needed. This is descriptive evidence, not a causal camera/population decomposition.

Draft protocol: `STEP_4_PLAN_DRAFT.md`. Audit review: `APPEARANCE_LABEL_AUDIT_REVIEW.md`. Next freeze the paper-faithful FundusAug comparator, global fitted transform, overlay-free fitted control, preservation measurements and exact transformation contract before classifier GPU launch.
