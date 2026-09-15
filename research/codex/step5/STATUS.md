# Step 5 status

**Updated:** September 15, 2026, 04:34 UTC  
**State:** strict source-only seed 0 running; frozen-feature diagnostics prepared at protocol level.

## Completed gates

- The design preflight reconstructed the original patient-separated cohorts: 11,372 BRSET fit images from 5,966 patients, 2,451 BRSET selection images from 1,279 patients and 732 mBRSET assessment images from 193 patients.
- The full allocated integrity audit decoded all 14,555 images with zero decode errors and found zero exact duplicate groups within or across roles/domains. This does not rule out near duplicates or cross-dataset patient identity.
- The CPU/provenance preflight passed syntax, manifest-hash, role-count, patient-separation, no-target-access and decoded-image-integrity checks.
- Allocated A40 smoke job 4402154 completed successfully. Six-update uninterrupted and stop/resume executions had zero model, EMA, optimizer or counter mismatches; no mBRSET assessment was accessed. The smoke is implementation evidence only.

## Active jobs

- **4402198:** full BRSET-only seed-0 training, running on one nonexclusive A40. It fits only BRSET and uses only BRSET validation for checkpoint and threshold selection.
- **4402199:** mBRSET assessment, dependency-locked with `afterok:4402198`. It cannot run after a failed training job.
- Measured runtime estimate is 8.25 hours for training, followed by roughly one validation-inference pass for assessment. Queueing and filesystem variability make this an estimate, not a promised completion time.

## Interpretation boundary

This run measures strict BRSET-to-mBRSET transfer. It does not compete fairly with B1 as a training strategy because B1 uses labeled mBRSET training images. It is a missing baseline and gap measurement, not the proposed contribution. The historically reused mBRSET test split will be described transparently and cannot be treated as untouched external validation.

## Next gate

In parallel with the source-only run, implement and smoke-test frozen B1 feature extraction. Reproduce the published GFP comparator and compare a matched global head with label-specific spatial evidence using fitting/validation data only. P1/P2/P3 training remains blocked unless the frozen spatial diagnostic passes its predeclared validation rule.
