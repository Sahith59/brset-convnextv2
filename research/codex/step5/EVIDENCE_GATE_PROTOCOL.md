# Step 5 evidence gate: do appearance variables explain B1 errors?

**Frozen before execution:** September 15, 2026  
**Scope:** mBRSET validation only; no test images, test predictions, model fitting or GPU use.

## Reason

Step 4 showed that global appearance matching did not pass the diagnostic replication gate. Before using the same degradation family inside a latent objective, test the missing premise directly: are B1's actual DR or ME errors associated with sharpness, brightness, contrast, illumination falloff or saturation?

This analysis is descriptive. An association would not prove that appearance causes an error. A weak association would be evidence against spending the next GPU batch on an appearance-conditioned mechanism.

## Inputs

- The fixed 725-image, 193-patient mBRSET validation split.
- Selection predictions and independently selected thresholds from B1 seeds 0, 1 and 2.
- The five Step-4 appearance measurements computed after the evaluation transform: bilinear 560 resize and center 512 crop.

The analysis must verify image order, labels and patient identity across all three prediction files and the frozen full-pool manifest. No identifiers or per-image outputs may be written to the public result.

## Analyses

For each label separately:

1. Calculate thresholded error and negative log-likelihood for each seed.
2. Measure Spearman association between each appearance variable and mean per-image negative log-likelihood. Apply Benjamini-Hochberg correction across the five variables for that label.
3. Use five-fold patient-grouped out-of-fold logistic regression to predict thresholded errors across all three seeds. Compare:
   - baseline: true label and seed indicators;
   - appearance model: baseline variables plus the five appearance variables.
4. Repeat the grouped analysis among positive images only, where the outcome is false negative.
5. Use a 2,000-draw patient-cluster bootstrap on the fixed out-of-fold predictions to describe AUROC and AUROC-difference uncertainty. This interval does not include uncertainty from choosing the analysis or training B1.

**Execution correction recorded before a result:** the first run stopped because one positive-only held-out fold contained no false negatives. A held-out fold does not need both outcomes when predictions are concatenated before AUROC calculation. The corrected check requires both outcomes in every fitting portion and in the combined out-of-fold cohort; the five patient-grouped folds, inputs, models and decision rule are unchanged.

## Frozen decision rule

Retain appearance-conditioned latent transition as the first novelty-facing GPU screen only if:

- for at least one label, the lower endpoint of the patient-bootstrap 95% interval for the appearance-model minus baseline error-prediction AUROC is above zero; and
- the point difference for the other label is at least `-0.01`; and
- the positive-image false-negative analysis is computable and its appearance model has AUROC above `0.60` for at least one label.

Otherwise remove the degradation action from the first proposed-method screen and pivot to a label-specific patch-evidence/shared-private feature mechanism. The `0.60` value is a practical screening threshold, not a clinical standard or significance level.

The strict source-only baseline and frozen-feature GFP comparator remain required under either outcome because they answer separate paper questions.
