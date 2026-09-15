# Step 5 direction revision after the appearance-error gate

**Date:** September 15, 2026  
**Status:** Replaces W1/W2/W3 as the first novelty-facing experiment. No Step-5 GPU work has launched.

## Why the original Step-5 idea is suspended

The initial Step-5 proposal was not ordinary blur augmentation: it would have used the applied transformation as a condition for predicting the clean latent representation. That is meaningfully different from Step 4. However, it still depends on the assumption that the measured appearance variables explain diagnostic failures. The validation-only evidence gate did not support that assumption.

The conclusion is narrow: the five global variables do not reliably identify current B1 errors. It does not invalidate CheXWorld or all acquisition modeling. CheXWorld addresses radiographs, uses a much larger pretraining regime, and combines local, global and domain objectives. It never established that our fitted BRSET-to-mBRSET action would improve binary DR/ME prediction.

## Evidence-led alternative

The next question is whether the global-average classification head is diluting small, label-specific retinal evidence. This is supported as a concern by:

- **Lesion-Aware Transformer, CVPR 2021:** weakly supervised lesion-region importance and diversity for DR grading.
- **Differentiable Patch Selection, CVPR 2021:** end-to-end selection of a small informative subset of high-resolution patches.
- **CCT-Net, ICCV 2021:** category-conditioned heatmaps and category-invariant refinement for cross-domain medical multi-disease classification.
- **Samba, NeurIPS 2024:** explicitly motivates recurrent patch modeling because the most severe lesion may occupy a small part of an image; reports a 4.1-point average accuracy improvement over VMamba in its cross-domain DR grading protocol.

These papers justify testing spatial evidence. They also prevent a claim that patch attention or category-specific pooling by itself is novel.

## Shared/private adaptation evidence and collision

Known-device residual adaptation is a plausible way to use BRSET while allowing a small mBRSET-specific correction:

- **Residual Adapters, NeurIPS 2017** learns compact domain-specific residual modules over a shared network.
- **Compact Feature Learning, CVPR 2019** combines shared and private representations and regularizes their relationship.
- **Exploiting Domain-Specific Features, NeurIPS 2021** argues that domain-specific information can improve domain generalization rather than always being discarded.
- **Customized Domain Adapters, ICCV 2025** uses customized adapters and a router for domain generalization.

Therefore generic shared/private encoders, adapters, routing or orthogonality are not safe novelty claims. Our historical private-encoder result also cannot be treated as a clean negative because its training budget and inference routing were not matched correctly, but repeating it unchanged would add little scientific value.

## Revised staged design

### 5A — strict source-only baseline

Train on BRSET train, select checkpoint and thresholds on BRSET validation, then perform one frozen mBRSET assessment. This is the only run that can be described as pure BRSET-to-mBRSET transfer.

### 5B — frozen-feature diagnostics

Using the existing B1 EMA checkpoint and no backbone retraining:

1. reproduce CVPR 2026 GFP on pooled features; and
2. compare a linear global-feature head with label-specific spatial top-k pooling heads on fitting/validation data.

The spatial diagnostic must use patient-grouped fitting/validation roles, tune its patch count only on validation, and remain test-free. Its purpose is to establish whether localized evidence contains information lost by global pooling.

### 5C — conditional mechanism screen

Only if the spatial diagnostic improves both-label mean AUROC or crosses the fixed F1 screen, freeze three controls:

- **P1:** label-specific patch evidence head, shared across devices;
- **P2:** known-device residual adapters with ordinary global pooling;
- **P3:** label-specific patch evidence plus known-device residual adaptation.

P3 must beat B1, P1 and P2 to support an interaction claim. Device identity is known from acquisition source; no learned inference router is used. This prevents the train/inference mismatch found in the historical routing experiment.

This screen can establish whether the combination is useful. It cannot by itself establish novelty, because both components have strong prior art. A final contribution statement requires an additional method distinction supported by the observed diagnostics and another closest-work audit.

## Generative world-model decision

A full generative model is now a lower-priority future branch. The evidence gate makes it less rational to synthesize global camera appearance, and generation still needs lesion-preservation validation. A world model would become relevant again if Dong identifies a specific model or if later spatial diagnostics show a lesion-local transformation problem that a localized latent predictor can address.

## Primary sources

1. Sun et al., Lesion-Aware Transformers for Diabetic Retinopathy Grading, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Sun_Lesion-Aware_Transformers_for_Diabetic_Retinopathy_Grading_CVPR_2021_paper.html
2. Cordonnier et al., Differentiable Patch Selection for Image Recognition, CVPR 2021: https://openaccess.thecvf.com/content/CVPR2021/html/Cordonnier_Differentiable_Patch_Selection_for_Image_Recognition_CVPR_2021_paper.html
3. Zhou et al., CCT-Net, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Zhou_CCT-Net_Category-Invariant_Cross-Domain_Transfer_for_Medical_Single-to-Multiple_Disease_Diagnosis_ICCV_2021_paper.html
4. Bi et al., Samba, NeurIPS 2024: https://proceedings.neurips.cc/paper_files/paper/2024/hash/8aa0c4d28021c0b273480f9e2aab83a6-Abstract-Conference.html
5. Rebuffi et al., Learning Multiple Visual Domains with Residual Adapters, NeurIPS 2017: https://proceedings.neurips.cc/paper_files/paper/2017/hash/e7b24b112a44fdd9ee93bdf998c6ca0e-Abstract.html
6. Liu et al., Compact Feature Learning for Multi-Domain Image Classification, CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Compact_Feature_Learning_for_Multi-Domain_Image_Classification_CVPR_2019_paper.html
7. Piratla et al., Exploiting Domain-Specific Features to Enhance Domain Generalization, NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/b0f2ad44d26e1a6f244201fe0fd864d1-Abstract.html
8. Ji et al., Customizing Domain Adapters for Domain Generalization, ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Ji_Customizing_Domain_Adapters_for_Domain_Generalization_ICCV_2025_paper.html
9. Pham et al., Post-training Feature Pruning for Fundus Images Classification, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Pham_Post-training_Feature_Pruning_for_Fundus_Images_Classification_CVPR_2026_paper.html

