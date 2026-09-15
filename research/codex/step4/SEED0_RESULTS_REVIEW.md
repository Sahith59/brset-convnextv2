# Step-4 seed-0 validation-screen review

## Completion and integrity

Slurm array `4397838` completed all three tasks successfully on September 15, 2026. A1, A2 and A3 each completed 5,775 updates and 369,600 training draws; their stderr logs are empty. Their contracts use the same full joint B1 fitting pool, seed-0 initialization, optimizer schedule, validation cohort and common augmentation pipeline. Only the source-image Step-4 operation differs. No A-arm assessment/test artifact exists.

The original summary job `4397839` failed before comparison because its no-test guard was incorrectly applied to the historical B1 reference, whose Step-2 directory legitimately contains its prior assessment. The guard was corrected to permit those pre-existing B1 files while reading only B1's validation summary; A1/A2/A3 remain required to be test-free. Allocated CPU rerun `4401233` completed with exit `0:0` and empty stderr. Input selection summaries and checkpoints were hash checked.

## Validation results

All values below use the selected EMA checkpoint and independently selected per-label validation threshold on the same 725-image mBRSET validation set.

| Arm | Meaning | DR F1 | DR delta vs B1 | DR AUROC | DR AUROC delta | ME F1 | ME delta vs B1 | ME AUROC | ME AUROC delta | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B1 | Natural joint baseline | 0.8703 | — | 0.9563 | — | 0.8489 | — | 0.9904 | — | Reference |
| A1 | FundusAug artifact component | 0.8772 | +0.0069 | 0.9530 | -0.0033 | 0.8462 | -0.0028 | 0.9904 | +0.0001 | Fail |
| A2 | Full target-fitted transform | 0.8739 | +0.0036 | 0.9558 | -0.0005 | 0.8485 | -0.0004 | 0.9906 | +0.0002 | Fail |
| A3 | Target-fitted transform without overlays | 0.8765 | +0.0062 | 0.9601 | +0.0038 | 0.8485 | -0.0004 | 0.9903 | -0.0001 | Fail |

The frozen replication gate required at least one label's F1 delta to be at least +0.01, the other label's F1 delta to be no worse than -0.01, and both AUROC deltas to be no worse than -0.01. No arm reached the +0.01 F1 requirement.

## Interpretation

This is a negative seed-0 screen for the tested augmentation mechanisms under the frozen rule. All arms produced small DR F1 increases, but ME was essentially unchanged or slightly lower. A3 also produced the best DR AUROC and average precision, so its overlay-free design is the most interesting signal, but a +0.0062 DR F1 change in one seed does not justify claiming improvement or launching automatic replication under the agreed rule.

The appearance result and diagnostic result therefore diverge: A2/A3 strongly reduced the five-statistic appearance distance, but that did not translate into a large validation improvement. This is evidence that matching global appearance statistics is insufficient as the proposed contribution.

These are tuned validation diagnostics, not test results, clinical benefit, statistical significance or a novelty claim. Step-4 A-arm test assessment remains unopened.

## Decision

Do not replicate A1, A2 or A3 automatically. Preserve the runs as negative evidence and use them to revise the research direction. The next method should introduce a mechanism that protects or predicts diagnostic content, rather than optimizing global appearance matching alone. A bounded CheXWorld-style latent-transition objective or a lesion-aware preservation constraint can be evaluated as a new, separately frozen step, but neither is supported until its protocol and falsification controls are defined.
