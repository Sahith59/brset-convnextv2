# Step 4 training-only appearance and label-composition audit

## Result

The known DR/ME label-composition difference does not explain most of the measured BRSET–mBRSET appearance gap. Standardizing both domains to the same pooled joint-label composition changed every gap by less than 0.05 mBRSET standard deviations and did not reverse any direction.

This supports a global training-only refit for the first classifier comparison. A separate label-standardized augmentation arm is not justified by this descriptive result.

## Evidence boundary

Allocated CPU job `4393429` completed with exit code `0:0` in 5:04 on `arctrdcn001`; stderr is empty. It measured all 11,372 BRSET and 3,402 mBRSET images in the original training splits after bilinear 560 × 560 resize and center 512 × 512 crop. No validation or test image was used.

Joint-label counts were BRSET `[10602, 22, 496, 252]` and mBRSET `[2591, 14, 520, 277]` for DR0/ME0, DR0/ME1, DR1/ME0 and DR1/ME1. Common-composition weights were the pooled proportions of these cells: `[0.8930, 0.0024, 0.0688, 0.0358]`. This keeps the rare DR0/ME1 cell rare rather than giving its few images artificial equal weight.

| Appearance statistic | Raw target−source gap, in target SD | Common-label-composition gap, in target SD | Change from standardization |
|---|---:|---:|---:|
| Sharpness | +2.0943 | +2.1209 | +0.0266 |
| Brightness | −0.3962 | −0.3693 | +0.0269 |
| Contrast | +1.3073 | +1.3436 | +0.0363 |
| Falloff | +0.1678 | +0.1701 | +0.0022 |
| Saturation | −0.4149 | −0.4587 | −0.0438 |

The strongest measured differences remain sharpness and contrast. Label standardization makes both slightly larger. Brightness becomes slightly less different, while saturation becomes slightly more different.

## Interpretation limits

This is a descriptive sensitivity analysis, not a causal decomposition. Matching the two recorded disease labels cannot separate camera effects from population, site, image quality, other retinal conditions or unmeasured differences. The five statistics are also global appearance summaries and do not establish lesion preservation.

The practical decision that changes below 0.05 target standard deviations are small was made after observing this audit and must not be presented as a preregistered statistical threshold. The safer conclusion is limited: known joint DR/ME composition does not reverse or substantially reshape the observed five-statistic directions.

## Step-4 decision

Proceed with a global target-training-only refit aligned to the classifier geometry. The first seed-0 classifier screen should compare existing B1 with a paper-faithful FundusAug component, the full fitted transform, and a fitted transform without synthetic spot/hole/halo overlays. The overlay-free arm tests whether the visually risky artificial lesions/artifacts are needed before building a more complex preservation constraint.

Authoritative aggregate output: `appearance_label_audit.json`. Independent integrity checks: `appearance_label_audit_verification.json`.

