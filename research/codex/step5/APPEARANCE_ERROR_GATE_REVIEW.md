# Step 5 appearance-error gate review

**Completed:** September 15, 2026  
**Allocated job:** 4402007 on `arctrdcn001`, exit `0:0`, 42 seconds  
**Scope:** three frozen B1 seeds on the 725-image, 193-patient mBRSET validation split; no test access and no model fitting.

## Result

| Label | Thresholded error rate across seeds | Appearance-model error AUROC change | Patient-bootstrap 95% interval | Positive-case false-negative appearance AUROC |
|---|---:|---:|---:|---:|
| DR | 0.0611 | +0.0108 | −0.0387 to +0.0605 | 0.5547 |
| ME | 0.0285 | +0.0472 | −0.0287 to +0.1284 | 0.4055 |

The appearance model used sharpness, brightness, contrast, illumination falloff and saturation in addition to true label and seed. The baseline error model used true label and seed only. Both were evaluated with five-fold patient-grouped out-of-fold predictions. The intervals resampled patients rather than images.

## Frozen decision

The evidence gate failed. Neither overall error-AUROC difference has a 95% interval whose lower endpoint is above zero, and neither positive-case false-negative appearance AUROC exceeds the frozen `0.60` screen. Therefore, do not run the planned action-conditioned W1/W2/W3 degradation experiment as the first Step-5 novelty screen.

This is evidence against the five global appearance variables as a useful explanation of current B1 errors. It is not proof that camera appearance never matters, because the analysis is observational, limited to five summaries, and uses development validation data.

## Execution correction

Job 4401999 stopped before producing a result because one positive-only held-out fold contained no false negatives. The original verifier unnecessarily required both outcomes in every held-out fold. The corrected job required both outcomes in each fitting portion and in the combined out-of-fold cohort. Cohort, folds, inputs, models, bootstrap and decision rule were unchanged. The failed job is retained in the log.

## Direction consequence

1. Keep the strict source-only baseline. It measures the true BRSET-only transfer gap.
2. Keep CVPR 2026 GFP. It tests whether B1's final representation contains harmful/redundant feature groups.
3. Add a frozen spatial-feature diagnostic: test whether label-specific patch evidence improves over global-average pooled B1 features without backbone retraining.
4. If patch evidence helps, screen label-specific patch pooling and known-device residual adapters with separate controls. If it does not, move to a stronger published retinal backbone/comparator rather than inventing a patch mechanism.
5. Do not claim novelty for generic patch attention, shared/private encoders or domain adapters; CVPR 2021 LAT and differentiable patch selection, ICCV 2021 CCT-Net, CVPR 2019 compact shared/private learning, NeurIPS 2017 residual adapters, NeurIPS 2024 Samba and ICCV 2025 customized domain adapters already occupy those ideas.

