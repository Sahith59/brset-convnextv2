# Step-2 final review

**Status:** complete on September 12, 2026.

## What was compared

- **B0, target-only:** train on 3,402 mBRSET training images.
- **B1, joint:** train on 11,372 BRSET plus 3,402 mBRSET training images in one mixed pool.
- **B2, source to target:** train first on 11,372 BRSET images and then fine-tune on 3,402 mBRSET images.

All arms used ConvNeXt V2 Tiny, the same total 5,775 optimizer updates, the same effective batch size 64, ordinary training augmentation and Mixup, and no fitted degradation, routing, or private encoder mechanism. Model selection and DR/ME threshold choice used the 725-image mBRSET validation split. The gated assessment used the same 732-image mBRSET test split for every arm and seed.

## Repeated results

| Seed | Arm | DR F1 | DR AUROC | ME F1 | ME AUROC |
|---:|---|---:|---:|---:|---:|
| 0 | B0 | 0.8000 | 0.9339 | 0.8571 | 0.9687 |
| 0 | B1 | 0.8312 | 0.9451 | 0.9060 | 0.9945 |
| 0 | B2 | 0.8041 | 0.9410 | 0.7805 | 0.9866 |
| 1 | B0 | 0.8051 | 0.9442 | 0.8000 | 0.9888 |
| 1 | B1 | 0.8310 | 0.9489 | 0.8224 | 0.9950 |
| 1 | B2 | 0.8239 | 0.9449 | 0.8430 | 0.9890 |
| 2 | B0 | 0.8264 | 0.9404 | 0.8148 | 0.9930 |
| 2 | B1 | 0.8393 | 0.9461 | 0.8571 | 0.9940 |
| 2 | B2 | 0.8137 | 0.9388 | 0.8319 | 0.9889 |

| Arm | DR F1, mean ± sample SD | DR AUROC | ME F1, mean ± sample SD | ME AUROC |
|---|---:|---:|---:|---:|
| B0 | 0.8105 ± 0.0140 | 0.9395 ± 0.0052 | 0.8240 ± 0.0297 | 0.9835 ± 0.0130 |
| B1 | **0.8338 ± 0.0048** | **0.9467 ± 0.0020** | **0.8619 ± 0.0420** | **0.9945 ± 0.0005** |
| B2 | 0.8139 ± 0.0099 | 0.9416 ± 0.0031 | 0.8184 ± 0.0333 | 0.9882 ± 0.0014 |

B1−B0 F1 differences were positive in all three seeds. Their means were +0.0233 for DR and +0.0379 for ME. This passes the prespecified engineering screen and makes B1 the appropriate reference for the next controls.

## What the evidence means

The ordinary joint baseline is stronger than the ordinary target-only and sequential baselines on average in this experiment. The repeated direction makes the result more credible than a single run. It also shows that a proposed method must beat a strong joint baseline rather than only beat target-only training.

This is not yet the paper's novelty. Three seeds provide a limited estimate of training variability. Per-seed patient bootstrap intervals condition on a trained model and do not include all training uncertainty. The test set was historically reused, so these values are development benchmark evidence rather than untouched external confirmation.

Equal optimizer updates also do not mean equal exposure. B0 draws about 369,600 target samples; natural B1 draws about 85,000 target samples because its pool is mostly BRSET; B2 draws about 184,832 target samples in its target phase. Consequently, Step 2 establishes which ordinary strategy is strongest but does not explain whether B1 improves because of domain mixing, class composition, or another effect.

## Verification and exceptions

All nine training runs and all three gated assessments completed. Independent jobs recomputed metrics, checked the common 732-image/193-patient cohort and ordering, verified labels, hashes, selected checkpoints, thresholds and exposure counters. Seed1 encountered one transient shared-filesystem read failure; no assessment was produced from the interrupted state. The affected file and all 3,402 target training images later decoded successfully on the same node, checkpoints passed contract checks, and the exact runs resumed to completion. The failure and recovery remain recorded.

## Next gate

Step 3 should freeze and test fixed-compute controls that independently change the source/target sampling ratio and the joint DR/ME label-stratum sampling. Candidate configurations must be selected with the mBRSET validation split. The reused test split should not be queried repeatedly while designing these controls. A fitted-degradation classifier experiment follows only after Step 3 identifies the strongest fair sampling baseline and clarifies what ordinary joint training is gaining from.
