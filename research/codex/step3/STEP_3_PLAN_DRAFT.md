# Step 3 draft: balance and exposure controls

Status: planning only. No Step-3 classifier job is authorized by this document or currently submitted. Freeze the executable supplement after baseline seeds 1 and 2 complete and the exact design-count artifact is reviewed.

## Purpose

Step 2 compares three useful training strategies, but equal optimizer updates do not make their data exposure equivalent. Step 3 determines whether a joint-training difference is sensitive to (1) how often source versus target images are sampled, (2) the joint DR/ME label distribution, or (3) additional target repetitions/optimization. This is causal-diagnostic work for a fair baseline; it is not the paper novelty.

All design and checkpoint choices use mBRSET validation. Do not assess each screening variant on the historically reused 732-image test set. Freeze the surviving comparison first, then perform a gated test assessment. Preserve image-level DR/ME co-occurrence by treating the four combinations DR0/ME0, DR0/ME1, DR1/ME0 and DR1/ME1 as strata; do not balance DR and ME independently.

## Stage 3A — evidence audit, no GPU training

1. Finish Step-2 seeds 1 and 2 and create per-seed selection-curve, exposure and test-metric tables.
2. Confirm actual source/target/DR/ME draws recorded in checkpoints against expected budgets.
3. Calculate image and patient counts for every domain × joint-label stratum.
4. Check whether all proposed sampling cells have enough unique images and quantify repetition. Reject a balance sampler that depends on an empty or extremely small cell.

Allocated count job 4376780 passed. BRSET strata DR0/ME0, DR0/ME1, DR1/ME0 and DR1/ME1 contain 10,602/22/496/252 images; mBRSET contains 2,591/14/520/277. Every cell is nonempty, but DR0/ME1 is small. Target-label matching would not justify treating repeated draws from this cell as new independent evidence; repetition reporting and a sensitivity check that groups all ME-positive images may be required before C2/C3 are frozen.

Expected target draws at 5,775 updates are 369,600 for B0, approximately 85,108 for natural B1, 184,800 for equal-domain joint, and 184,832 for B2. Thus B1's seed-0 advantage cannot be explained by receiving more target draws than B0; it received fewer. This does not isolate why source images help.

## Stage 3B — fixed-compute sampling controls

Use the joint B1 architecture, initialization, loss, transforms, 5,775 updates, effective batch 64 and validation rule. Change only the stateless sampler.

| Arm | Domain marginal | Label-stratum sampling | Question |
|---|---|---|---|
| B1 natural, existing | Proportional to pool size | Natural | Reference |
| C1 equal-domain | 50% BRSET / 50% mBRSET | Natural within each domain | Is the result sensitive to target/source frequency? |
| C2 target-label-matched | Same natural domain marginal as B1 | Within each domain, sample four strata at mBRSET-training proportions | Is the result sensitive to the label/co-occurrence distribution? |
| C3 equal-domain + target-label-matched | 50% / 50% | Four target label-stratum proportions within each domain | Is there a domain-ratio × label-balance interaction? |

Each microbatch should have a deterministic declared composition where feasible. Keep Mixup implementation and all non-sampler random streams identical across matched arms. Report actual unique images, repeats, source/target draws, label draws and compute time. The target-label design is conditional on every domain having adequate support for all four strata; `design_counts.json` decides feasibility.

Screen C1–C3 with seed 0 on validation only. Continue a control only if it changes DR F1 by at least 0.01, materially changes ME, or is necessary to interpret B1. Replicate any interpretation-critical control with seeds 1 and 2. A small effect supports retaining natural B1 for simplicity; it is not a negative research outcome.

## Stage 3C — exposure control

No single experiment can hold optimizer updates, total image draws, target draws and added source information all constant simultaneously. Report this tradeoff explicitly.

The primary exposure experiment, conditional on B1 remaining competitive, is an equal-domain joint run with 11,550 updates: 32 target and 32 source images per effective update. It matches B0's 369,600 target draws while adding 369,600 source draws. Because it doubles updates and compute, pair it with a declared longer-compute target-only control or interpret it only as a resource-unmatched upper bound. Do not attribute its difference solely to source information.

Before launching the long control, inspect all three seeds' final-quarter validation curves. If ordinary runs have plateaued and fixed-compute C1–C3 answer the sampling question, skip the costly extension. If curves are still improving or the source effect remains ambiguous, freeze a resumable 11,550-update recipe and matched compute comparator. The 12-hour job limit requires an explicit checkpoint/resume plan.

## Decision boundary

Step 3 is complete when the exposure ledger is verified, fixed-compute sampler controls needed for interpretation are replicated, and any justified long-exposure comparison is resolved. Select the simplest strong baseline for Step 4 using cross-seed validation evidence, with DR primary and ME explicitly protected. Do not use the test set to choose C1, C2, C3, or the long-run design.

Then Step 4 compares ordinary augmentation, a faithful published FundusAug component and FIT-only fitted degradation under the selected baseline. The degradation mechanism must improve diagnosis across seeds and pass a diagnostic-damage/preservation check; appearance distance alone is insufficient.
