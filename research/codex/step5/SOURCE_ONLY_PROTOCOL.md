# Strict source-only baseline protocol

**Version:** 1.0, frozen September 15, 2026 before training  
**Purpose:** quantify pure BRSET-to-mBRSET transfer without target-domain fitting or selection.

## Cohorts

- Fit: original BRSET train, 11,372 images / 5,966 patients.
- Selection: original BRSET validation, 2,451 images / 1,279 patients.
- Assessment: original mBRSET test, 732 images / 193 patients.
- BRSET test is unused.
- mBRSET train and validation are unused.

The private manifest SHA256 is `635d3f47f07ae6c6dc656f8e32036e9622d32ab0ab09afbda535d010389ca457`. A new decoded-image and exact-duplicate gate covering these three cohorts must pass before GPU smoke or training.

## Training and selection

Match Step-2 B1 wherever the source-only data boundary permits:

- ConvNeXtV2-Large and the same seed-0 initialized head;
- 5,775 optimizer updates, effective batch 64;
- bf16, AdamW, learning rate `3e-5`, weight decay `0.1`, 693-update warm-up and cosine decay;
- focal loss, Mixup `0.2`, ordinary resize/crop/flip/rotation/color augmentation;
- EMA decay `0.999` and EMA-only four-flip evaluation;
- checkpoint selection by mean DR/ME AUROC on BRSET validation at the same 231-update intervals;
- independent DR and ME thresholds chosen on BRSET validation using the fixed 0.001 grid and tie rule.

No mBRSET image, label, statistic or prediction may be accessed by training, checkpoint selection, threshold selection or hyperparameter choice.

## Assessment

After training, provenance checks and threshold freezing, run one assessment on the historically reused mBRSET test split. Report:

- positive-class F1 at the BRSET-selected threshold;
- AUROC and average precision;
- precision, sensitivity and specificity;
- class-macro F1, confusion matrix and F1 at threshold 0.5.

The result is a strict source-only benchmark but not untouched external validation. Seed-1/2 replication is conditional on whether source-only transfer becomes central to the final paper claim.

## Required gates

1. Design/cohort preflight and manifest hash.
2. Full decoded-image and exact cross-role duplicate audit.
3. CPU unit/provenance tests.
4. One allocated-A40 six-update smoke with full BRSET validation inference.
5. Stop/resume equivalence for model, EMA, optimizer and counters.
6. Full seed-0 training; automatic assessment only after successful completion and verification.

