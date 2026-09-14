# Step 4 protocol: target-calibrated appearance augmentation

## Purpose in simple words

Step 4 asks whether changing BRSET training images to reflect realistic mBRSET acquisition variation helps the classifier recognize real mBRSET disease. The existing appearance-distance result only says that five global image summaries moved closer. It does not say that lesions were preserved or that classification improved.

Natural joint training B1 is the fixed reference. Step 3 showed that forcing a 50/50 source-target sampler was not a clearly stronger baseline, so Step 4 does not carry that extra mechanism forward.

## Scientific question

Under the same B1 data, model, optimizer, number of image draws and validation boundary, does a target-informed fundus degradation policy improve image-level any-DR and ME classification beyond ordinary augmentation and a release-range-faithful FundusAug artifact component? Diagnostic preservation is a separate required gate before any final method claim.

## Why an audit comes before a new training run

The original fitted transform matches global BRSET and mBRSET appearance summaries. The domains also have different DR/ME label proportions, and lesions can change sharpness, contrast and brightness. A global fit could therefore absorb disease-composition differences into what is called an acquisition transformation.

The first allocated Step-4 analysis measures all five appearance statistics:

1. over each domain's complete original training pool;
2. within each of the four joint DR/ME label cells; and
3. after both domains are standardized to the same pooled joint-label composition.

This is descriptive sensitivity analysis. Matching known DR/ME labels does not isolate camera effects from population, site, quality or unmeasured disease differences.

## Staged comparison

### Stage 4A — evidence and implementation gate

- Complete the joint-label appearance audit on allocated CPU compute.
- Freeze the exact classifier geometry: bilinear resize to 560, then 512 crop.
- Refit any target-informed transform using original training images only. Do not access mBRSET validation or test images during fitting.
- Implement a release-range-faithful FundusAug artifact component independently from the paper description and inspected release, with upstream commit and adaptation choices recorded. The upstream repository inspected on September 14 has no visible license file, so do not copy or redistribute its source.
- Define deterministic random streams, application order, probabilities, exposure ledgers and interruption/resume behavior.
- Inspect representative transformed images for implementation errors and grossly destructive behavior before classifier training. This qualitative gate does not establish diagnostic preservation.

The refit uses two deterministic, patient-disjoint BRSET training subsets: 96 calibration images and 96 validation images, with no patient appearing in both. Target means and scales come from all 3,402 mBRSET training images measured in the completed appearance audit. A fixed 256-candidate random search is run separately for the full and overlay-free operator spaces. Each candidate is evaluated with two stochastic transform draws per calibration image; the selected candidate is then checked with three unseen transform seeds on the held-out source subset. These counts are a compute-conscious parameter-fitting design, not a claim that 96 images characterize every source subgroup.

The full fitted search varies blur, illumination, bright spots, dark holes, halo, sensor noise and color gains. The overlay-free search fixes spot, hole and halo terms to zero and refits the remaining terms rather than simply deleting overlays after fitting. Both use the same five-statistic standardized squared-mean objective as the advisor report, recalculated under 560-resize/512-center-crop geometry.

The A1 comparator is specifically the **FundusAug artifact component adapted to 512 pixels**, not full GDRNet. It independently implements the paper/release ranges for sharpness, halo, hole, spot and blur, each with probability 0.5. The ordinary B1 crop, horizontal flip, rotation and color jitter remain common to every arm. The upstream release additionally uses vertical flip, a much broader color jitter and a 256-to-224 geometry; those differences are intentionally not imported because they would change several baseline variables at once.

### Stage 4B — seed-0 validation screen

The initial comparison reuses the existing B1 seed-0 validation result and trains only frozen new arms:

| Arm | Additional source-image transform | Role |
|---|---|---|
| B1 | None beyond the ordinary frozen pipeline | Existing reference |
| A1 | Independently implemented FundusAug artifact component using published/release ranges | Published augmentation comparator; not full GDRNet |
| A2 | Training-only global target-fitted parametric degradation | Direct test of the degradation work sent to Dong |
| A3 | Global fitted transform without synthetic spot/hole/halo overlays | Test whether visually risky overlays are necessary |

Additional transformations apply to BRSET draws only; labeled mBRSET draws keep the ordinary B1 pipeline. All arms retain the same natural B1 sampler, 5,775 optimizer updates, effective batch 64, focal loss, Mixup, EMA selection and 725-image mBRSET validation cohort. No Step-4 test assessment is used for screening.

The frozen training order is: bilinear resize to 560, random 512 crop, the arm-specific source operation for BRSET only, then the common horizontal flip, rotation and color jitter. Target images skip the source operation. This places the fitted operation at the same 512-pixel geometry used during parameter fitting while preserving the baseline random-operation order for the common pipeline.

The completed Stage-4A audit found no direction changes and at most a 0.0438-target-SD change after joint-label standardization. Therefore use a global training-only target refit and do not add a label-standardized classifier arm. This is a post-audit practical decision, not a preregistered statistical threshold.

Prioritize an arm for replication only if one label's validation F1 improves by at least 0.01, the other label is not worse by more than 0.01, and neither label's AUROC is worse by more than 0.01. This is an engineering screen, not a clinical or significance threshold.

### Stage 4C — diagnostic-preservation validation and constrained mechanism

If Stage 4B identifies a promising fitted arm, first validate a preservation signal against lesion/vessel-sensitive checks and qualified review when feasible. Only then compare the selected fitted transform with:

- a simple lower-severity version;
- a preservation-constrained version that rejects or reduces transformations exceeding a frozen diagnostic-damage budget; and
- a matched random-rejection control with the same acceptance rate and image exposure.

The matched rejection control is necessary because selecting fewer transformed views can itself change training. Classifier prediction stability or saliency alone is not clinical proof. Use lesion/vessel-sensitive downstream checks and qualified review when feasible, following the evaluation dimensions emphasized by EyeBench-V2.

### Stage 4D — replication and paper gate

Replicate only the interpretation-critical candidate at seeds 1 and 2. A method-facing result requires a consistent cross-seed validation effect, an ablation isolating the preservation constraint, exact resource accounting, and a closest-work comparison. Only then perform a gated benchmark assessment and decide whether the evidence supports a paper contribution.

## Current novelty boundary

Already occupied ideas include fundus visual/artifact augmentation and strong/weak consistency (GDRNet), grade-conditioned diffusion augmentation (DG-ADR), structure-preserving retinal translation/enhancement, clinical downstream evaluation of retinal enhancement (EyeBench-V2), and saliency-preserving medical-image augmentation (MedDiffuseMix, a 2026 preprint). Therefore none of the following is novel by itself: adding blur/spots, fitting global appearance means, preserving saliency, using consistency, or combining existing components.

The current candidate distinction is narrower: a low-compute, label-composition-aware target-device calibration of a parametric acquisition transform, fitted without paired camera images or a generative model and coupled to a validated diagnostic-preservation budget for supervised handheld-camera transfer. This is a hypothesis, not a novelty claim. The literature audit and experiments may reject it.

## Falsification and pivot rules

- If A2 does not improve over B1 and FundusAug, the current fitted degradation is not an effective classifier method.
- If label standardization barely changes the fitted target or A2/A3 behavior, do not claim disease-composition correction as a contribution.
- If the preservation signal does not agree with lesion/vessel-sensitive checks or qualified review, do not use it as evidence of diagnostic preservation.
- If the constrained arm does not beat severity and matched-rejection controls, do not claim the constraint is responsible.
- If no augmentation arm repeats, retain the result as a rigorous negative/sensitivity study and pivot to a different mechanism rather than renaming the same components.

## Primary closest sources checked

- Che et al., *Towards Generalizable Diabetic Retinopathy Grading in Unseen Domains* (MICCAI 2023): https://arxiv.org/html/2307.04378v3
- Official GDRNet repository, inspected commit `10f339748de69bd2f4b1bec49858eef95082679c`: https://github.com/chehx/DGDR
- Chokuwa and Khan, *Divergent Domains, Convergent Grading* (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/papers/Chokuwa_Divergent_Domains_Convergent_Grading_Enhancing_Generalization_in_Diabetic_Retinopathy_Grading_WACV_2025_paper.pdf
- Vasa et al., *Context-Aware Optimal Transport Learning for Retinal Fundus Image Enhancement* (WACV 2025): https://openaccess.thecvf.com/content/WACV2025/papers/Vasa_Context-Aware_Optimal_Transport_Learning_for_Retinal_Fundus_Image_Enhancement_WACV_2025_paper.pdf
- Dong et al., *Bridging Restoration and Diagnosis: A Comprehensive Benchmark for Retinal Fundus Enhancement* (WACV Workshops 2026): https://openaccess.thecvf.com/content/WACV2026W/P2P/html/Dong_Bridging_Restoration_and_Diagnosis_A_Comprehensive_Benchmark_for_Retinal_Fundus_WACVW_2026_paper.html
- Kumar et al., *MedDiffuseMix* (arXiv preprint, 2026): https://arxiv.org/abs/2606.28419
