# Step 3 final review: balance and exposure controls

## Decision

Step 3 is complete. Retain **B1 natural joint training** as the reference for Step 4. Do not launch the conditional 11,550-update exposure experiment.

This is a validation-based engineering choice. It is not a statistical-significance, clinical, external-generalization, or novelty claim.

## Completion and integrity evidence

- C1 seed-1 and seed-2 replication wave `4385146` completed with exit code `0:0` in 7:24:41 on two nonexclusive A40 nodes.
- Verification and report job `4385147` completed with exit code `0:0`.
- All replication stderr files are empty.
- Contracts, initialization and checkpoint provenance, exact exposure ledgers, finite 725-image validation metrics, and the absence of Step-3 test assessment artifacts passed the frozen verifier.
- The authoritative replicated values are in `c1_validation_three_seeds.json`.

## What the replicated comparison shows

C1 changes only the training sampler: source and target each receive 50% of the 369,600 image draws. B1 keeps the natural pooled ratio, which gives approximately 23% of draws to mBRSET. Model, data pools, optimizer updates, loss, ordinary augmentation, checkpoint selection, and validation cohort remain fixed.

| Outcome | B1 mean ± sample SD | C1 mean ± sample SD | Mean paired C1−B1 | Positive seeds |
|---|---:|---:|---:|---:|
| DR F1 | 0.8703 ± 0.0025 | 0.8738 ± 0.0076 | +0.0035 | 2/3 |
| DR AUROC | 0.9586 ± 0.0054 | 0.9515 ± 0.0032 | −0.0071 | 0/3 |
| DR average precision | 0.9353 ± 0.0027 | 0.9314 ± 0.0038 | −0.0039 | 0/3 |
| ME F1 | 0.8457 ± 0.0034 | 0.8536 ± 0.0089 | +0.0079 | 3/3 |
| ME AUROC | 0.9889 ± 0.0014 | 0.9880 ± 0.0016 | −0.0009 | 0/3 |
| ME average precision | 0.9154 ± 0.0084 | 0.9176 ± 0.0084 | +0.0022 | 3/3 |

The seed-0 ME F1 change that triggered replication shrank from +0.0147 to a three-seed mean of +0.0079. DR F1 changed direction across seeds. C1 also reduced mean AUROC for both labels. These are small metric tradeoffs, not evidence that equal-domain sampling is a stronger general reference.

## Why the long exposure experiment is not justified

The protocol made the 11,550-update comparison conditional on evidence that ordinary 5,775-update runs were still improving or that the source effect remained ambiguous after fixed-compute controls. Neither condition is present.

For B1 and C1 at all three seeds, the best mean-DR/ME-AUROC checkpoint occurred at update 2,541–3,234, before the final quarter. From the first final-quarter checkpoint to update 5,775, the selection score decreased in all six runs by 0.0014–0.0062. More updates would therefore test an already declining schedule rather than repair visible undertraining. C1 already raised target exposure from approximately 85,108 to 184,800 draws at fixed total compute without a clear overall advantage.

## What Step 3 established

1. B1's Step-2 advantage over target-only training cannot be explained by receiving more target-image draws: B1 receives far fewer target draws than B0.
2. Forcing equal source/target frequency has only small, metric-dependent effects across three validation seeds.
3. Seed-0 target-label matching and its interaction with equal-domain sampling did not cross the fixed 0.01 F1 replication screen. The small DR0/ME1 cells also make aggressive label matching repetition-heavy.
4. The simplest strong sampling reference remains B1. Step 4 can now isolate appearance augmentation without carrying an unnecessary balancing mechanism into the method.

## Next experiment

Step 4 should compare, under the frozen B1 recipe and validation boundary:

1. B1 with its existing ordinary augmentation;
2. B1 with a faithful implementation of the selected published fundus augmentation component;
3. B1 with the FIT-only fitted degradation transform;
4. only if the fitted transform is promising, an ablation that adds the diagnostic-preservation constraint.

Before launching, freeze exact operators, probabilities, parameter ranges, FIT-only estimation data, lesion-preservation measurements, compute budget, and a falsification rule. The fitted degradation result is currently an appearance-distance finding only.
