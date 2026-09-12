# Step-3 fixed-compute sampler protocol

Status: frozen for implementation verification and seed-0 validation screening on September 12, 2026.

## Fixed factors

All controls inherit the completed B1 recipe: ConvNeXt V2 Tiny, seed-0 initialization, 5,775 optimizer updates, effective batch 64, focal loss, ordinary image augmentation, Mixup, AdamW schedule, EMA, four-flip inference, checkpoint selection by mean DR/ME validation AUROC, and validation-only threshold selection. The fit pool remains 11,372 BRSET plus 3,402 mBRSET images. No degradation, routing, private encoder, test assessment or other mechanism is introduced.

## Changed factor

Only the deterministic training sampler changes.

| Arm | Source/target quota | Within-domain joint-label quota |
|---|---|---|
| C1 equal domain | 50%/50% | Natural |
| C2 target label | Natural pool ratio | mBRSET train proportions for DR0/ME0, DR0/ME1, DR1/ME0, DR1/ME1 |
| C3 equal domain plus target label | 50%/50% | mBRSET train proportions for all four strata |

Integer quotas use the largest-remainder rule over all 369,600 image draws. A cumulative-deficit schedule spreads the exact quotas through training. C1 and C3 contain exactly eight source and eight target images in every 16-image microbatch. Within each sampling cell, deterministic shuffled passes use every image before repeating it. Resume positions reproduce the suffix of an uninterrupted schedule exactly.

## Evidence and decision rule

Each run records domain and joint-label draws, unique images and repeat draws. Seed-0 screening uses only the 725-image mBRSET validation split. Compare with the existing seed-0 B1 validation result. Replicate an interpretation-critical control at seeds 1 and 2 if it changes DR F1 by at least 0.01, materially changes ME, or is required to interpret B1. Do not assess C1–C3 on the historically reused test split during screening.

The DR0/ME1 source and target cells contain only 22 and 14 unique images. Their repetitions are reported as sampling repetitions, never as additional independent evidence. Before using a target-label-matched arm as the paper baseline, run a sensitivity analysis that merges the two ME-positive strata or otherwise reduces reliance on this small discordant cell.
