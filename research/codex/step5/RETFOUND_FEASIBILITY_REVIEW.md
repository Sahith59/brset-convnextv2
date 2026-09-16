# Protocol-matched RETFound feasibility and launch review

**Status:** the evidence and engineering gates passed; seed 0 is running. No model-performance result exists yet.

## Why the old RETFound result cannot be reused

The companion repository contains useful official RETFound code and a pretrained retinal ViT-Large weight, but its previous experiments answer a different question. They predict referable DR, omit ME, use a different mBRSET split for 2,262 of 4,859 shared files, assign a different binary DR label to 271 files, use 224-pixel inputs, and follow a different training and selection procedure. Reusing those numbers would create an invalid comparison with B1.

## Matched comparator

The new comparator uses the exact B1 evidence boundary:

- fit on 11,372 BRSET and 3,402 mBRSET training images;
- select checkpoints and DR/ME thresholds on 725 mBRSET validation images from 193 patients;
- predict any DR and ME with two sigmoid outputs;
- resize to 560 pixels and crop to 512 pixels;
- use 5,775 optimizer updates and effective batch size 64;
- use AdamW, peak learning rate 3e-5, 693 warmup updates, weight decay 0.1, gradient clipping at 1.0, EMA 0.999, focal loss, Mixup 0.2, and label smoothing 0.1;
- evaluate four deterministic flips and select by mean DR/ME validation AUROC;
- forbid the 732-image assessment split during development.

RETFound is a stronger published retinal backbone comparator. Better performance would strengthen the baseline evidence but would not establish our novelty.

## Gate evidence

| Check | Evidence | Result |
|---|---|---|
| Cohort and label preflight | Job 4410136 | Pass: 14,774 fit images, 725 validation images, two labels, no assessment access |
| Weight loading | Job 4410137 | Pass: 100% backbone-parameter coverage; only the new two-output head is missing |
| Real-image feasibility | Job 4410137 | Pass: four fit microsteps, finite loss/gradient, 7.02 GiB peak allocation |
| Runtime estimate | Job 4410137 | Pass: 7.01 hours estimated, below the frozen 60-hour ceiling |
| Exact interruption/resume | Job 4410171 | Pass: six updates, zero state mismatch, no assessment access |
| Full seed-0 early health | Job 4411081 | Pass so far: update 1 finite; empty error log |

## Corrected failures and why they do not affect the science

| Job/event | Failure | Correction and treatment |
|---|---|---|
| 4410101 | NumPy Boolean could not be serialized to JSON after all checks ran | Cast values to native Boolean and reran preflight; 4410136 is authoritative |
| 4410102 | Dependent smoke was cancelled after its prerequisite failed | No scientific output; replaced by the final gated chain |
| 4410105 | Accidentally submitted to the wrong partition | Cancelled before execution and resubmitted to qTRDGPU/A40 |
| 4410107 | Gradient-checkpoint warning made the runtime interpretation unreliable | Required input gradients and verified 100% backbone gradient coverage; result excluded |
| 4410109 / 4410120 | Valid calibration runs at microbatch 1 and 16 | Microbatch 16 and accumulation 4 selected because it matches B1 and is feasible |
| 4410138 | Full and resumed states differed slightly under a non-strict deterministic setting | Enabled deterministic algorithms; discarded output; 4410171 passed exactly |

These are implementation or scheduling failures. They produced no scientific comparison and are excluded from the model result. The negative GFP and spatial findings reported earlier came from separate completed and verified runs.

## Current decision

Job 4411081 runs one seed on one nonexclusive A40. Its first update had loss 0.17327 and finite gradient norm 4.175. This confirms that the final frozen implementation is training. It does not tell us whether RETFound is better than B1.

When seed 0 finishes, compare it with B1 on the same mBRSET validation split. Replicate only if the frozen material-gain rule is met. Do not access the assessment split to rescue a failed validation comparison. The current novelty status remains **unproven**.
