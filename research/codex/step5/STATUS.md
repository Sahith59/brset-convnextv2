# Step 5 status

**Updated:** September 15, 2026
**State:** strict source-only seed 0 completed and independently verified; frozen-feature diagnostics are next.

## Completed gates

- The design preflight reconstructed the original patient-separated cohorts: 11,372 BRSET fit images from 5,966 patients, 2,451 BRSET selection images from 1,279 patients and 732 mBRSET assessment images from 193 patients.
- The full allocated integrity audit decoded all 14,555 images with zero decode errors and found zero exact duplicate groups within or across roles/domains. This does not rule out near duplicates or cross-dataset patient identity.
- The CPU/provenance preflight passed syntax, manifest-hash, role-count, patient-separation, no-target-access and decoded-image-integrity checks.
- Allocated A40 smoke job 4402154 completed successfully. Six-update uninterrupted and stop/resume executions had zero model, EMA, optimizer or counter mismatches; no mBRSET assessment was accessed. The smoke is implementation evidence only.

## Completed source-only seed 0

- **4402198:** full BRSET-only seed-0 training completed in 8:17:10 on one nonexclusive A40.
- **4402199:** dependency-locked mBRSET assessment completed in 1:03 after successful training.
- **4404962:** independent recomputation/provenance verification passed.
- The BRSET-selected checkpoint was update 2,310, with DR/ME thresholds 0.399/0.451.
- On 732 mBRSET assessment images, DR F1/AUROC were 0.7218/0.9100 and ME F1/AUROC were 0.6972/0.9734. DR and ME sensitivity were 0.6038 and 0.6032.

## Interpretation boundary

This run measures strict BRSET-to-mBRSET transfer. The lower positive-class F1 and sensitivity show a meaningful source-to-target gap for seed 0 even though threshold-free AUROC remains high. It does not compete fairly with B1 as a training-resource strategy because B1 uses labeled mBRSET training images. It is a missing baseline and gap measurement, not the proposed contribution. The historically reused mBRSET test split will be described transparently and cannot be treated as untouched external validation. One seed does not measure training variability.

## Next gate

Implement and smoke-test frozen B1 feature extraction. Reproduce the published GFP comparator and compare a matched global head with label-specific spatial evidence using fitting/validation data only. P1/P2/P3 training remains blocked unless the frozen spatial diagnostic passes its predeclared validation rule.
