> SUPERSEDED by report/PLAN_v2.md (8 Oct 2026). Kept for the record of the first pivot draft.

# Paper plan: learned device translation for cross-camera retinal screening

Written 8 October 2026, in response to the review that the work so far reads as
preprocessing rather than a contribution. That judgement is correct and this
document replaces the previous direction.

---

## 1. Why the previous direction could not have worked

Measured 8 October, after the review. At matched best-cutoff on the same
732-image target test set:

| | DR F1 |
|---|---|
| Target-trained, best of six recipes | 0.8456 |
| Target-trained, mean of six | 0.8271 |
| Pooling BRSET + all 3,402 mBRSET (Experiment 0) | 0.8455 |
| Pooling, 6-model ensemble | 0.8581 |

**Pooling already equals the best model obtainable by training directly on
target data.** The setting is saturated. Five successive architectures all
landed inside the 0.0199 seed-noise floor because there was no headroom to
compete for, not because the ideas were bad. Even ensembling, which almost
always helps, gains only 0.0126 and stays inside the floor.

The consequence for planning: **no architecture-side intervention evaluated at
full target data can produce a publishable number.** The axis has to change.

## 2. The axis that does have headroom

Real handheld images are expensive to collect and label. BRSET images are
already paid for. So the question is not "can a better architecture beat
pooling" but:

> **How many real target labels does pooling actually need, and can synthetic
> target images replace them?**

That reframes the saturation result from an obstacle into the paper's premise.
Saturation at 3,402 images is the thing to be explained and exploited, and the
interesting regime is everything to the left of it.

## 3. What the literature already covers, and the gap

| Work | What it does | Why it is not this |
|---|---|---|
| cofe-Net, GDRNet (FundusAug) | hand-crafted fundus degradation | assumes degradation means blur; measured here to move AWAY from the target on 4 of 5 statistics |
| CycleGAN SLO to colour fundus, for glaucoma | unpaired GAN translation, downstream AUC 0.89 to 0.94 | modality translation, not device class; explicitly names diffusion as unexplored future work |
| Camera to cSLO translation, Sci Rep 2026 | paired translation, self-attention, FMSR metric | paired same-patient data; evaluated on realism and structure, no downstream disease task |
| EyeBench, "Bridging Restoration and Diagnosis", WACV 2026 | benchmark arguing enhancement must be judged on diagnosis | a call for the evaluation this paper would supply |
| Fundus IQA/enhancement systematic review 2025 | surveys the field | states diffusion for UNPAIRED fundus enhancement "remains limited" |

**The gap, stated precisely.** No work performs *device-class* translation
(clinical tabletop to handheld smartphone-based camera) on *unpaired* data,
with a *lesion-preservation guarantee*, evaluated on *downstream disease
detection* as a function of *target label budget*. Each of those five
qualifiers is satisfied individually somewhere; the combination is open, and it
is the combination a screening programme actually cares about.

## 4. Claims the paper will make

1. **Hand-crafted degradation is mis-specified for this device pair.** The gap
   is noise and contrast, not blur. GDRNet's published settings score a
   normalised appearance distance of 7.75 against untouched BRSET's 5.13, so
   they are worse than doing nothing. *(Already measured.)*
2. **Learned translation fixes the appearance gap while preserving lesions.**
   Verified two ways: appearance distance to the measured target, and a
   lesion-preservation score under a classifier trained only on real target
   data. Perceptual metrics such as FID are reported but not relied on.
3. **Headline. Synthetic target images substitute for real target labels.**
   Joint training with synthesis reaches full-pooling performance at K real
   target labels, K far below 3,402. The claim is a curve, not a single number.
4. **Every comparison is made against a measured noise floor.** 0.0199 DR F1
   and 9 patients, from seed-only repetition. The saturation at full data is
   reported rather than hidden, and it explains the negative results.

Claims 1 and 4 are done. Claims 2 and 3 are the work.

## 5. Experiment grid

**Axis A, real target budget:** 0, 250, 500, 1000, 2000, 3402.

**Axis B, what is added to BRSET:**
- B0 nothing (the baseline curve)
- B1 GDRNet published hand-crafted degradation
- B2 fitted hand-crafted degradation *(parameters already fitted, distance 1.36)*
- B3 CycleGAN or CUT unpaired translation *(the standard baseline reviewers will demand)*
- B4 diffusion translation with lesion preservation *(the proposed method)*

**Seeds:** 2 per cell minimum, 3 for the headline cells. Nothing is reported
that does not clear the noise floor.

Not every cell is needed. Priority order: the B0 curve, then B4 and B3 at
budgets 250/500/1000, then B1 and B2 at those budgets as the hand-crafted
comparison, then full-budget cells to demonstrate convergence.

## 6. Method: a dual-domain translator, after DuDoNet

Dr. Ye pointed to **DuDoNet** (Lin et al., CVPR 2019), which is a CT metal-artifact
method, so the relevance is structural rather than topical. Its thesis is that
restoration needs TWO domains at once, the sinogram domain where the acquisition
physics lives and the image domain where appearance lives, joined by a
differentiable physical operator (Radon inversion) so the pair trains end to end.
Neither domain alone suffices: sinogram-only leaves secondary artifacts,
image-only has no access to the acquisition process.

That argument transfers exactly, and our own measurements already prove both
halves of it for this problem:

- **Physics alone is insufficient.** Fitting the nine-parameter acquisition
  chain to the measured mBRSET statistics cut normalised distance from 5.13 to
  1.36, yet the images still do not look like mBRSET. Framing and hue dominate
  perceptually and are invisible to those five statistics.
- **Unconstrained statistic-matching is actively unsafe.** Making the chain
  differentiable and fitting by gradient descent WITHOUT bounds drives spot and
  hole amplitude to 2.98 and 1.95 on a [0,1] image. The optimiser hits the
  target statistics by painting large blobs: numerically excellent, visually
  destroyed. Bounded to physically plausible ranges it reaches a 91.8 percent
  distance reduction, with spot and hole amplitude both pinned at their ceiling
  and contrast and saturation still short. **The physical model is exhausted and
  the gap remains.** That residual is precisely what the learned stage is for.

**Architecture.** Stage one is `PhysicalAcquisition` (script 72), the nine
interpretable parameters of the fitted chain, now differentiable and bounded.
Stage two is a learned residual refiner that closes what physics cannot, trained
adversarially against real mBRSET. The two train jointly, which is the DuDoNet
construction, with the bounded physics acting as the structural prior that keeps
the generator honest.

**Why this preserves labels.** A translated BRSET image keeps its BRSET label,
so a generator that erases a microaneurysm silently corrupts the training set.
The acquisition stage is photometric and cannot move anatomy at all. The refiner
is constrained to a bounded residual on top of it. Preservation is then
*measured*, not assumed, using a classifier trained only on real target data.

**Baselines.** No augmentation; GDRNet published settings; our fitted
hand-crafted chain; CycleGAN/CUT unpaired translation. The last is what the
closest prior work used and what reviewers will demand.

## 7. Venue and timeline: ISBI 2027

**Target: ISBI 2027 four-page, deadline 26 October 2026.** MIDL 2027 (4 December)
is the fallback and the venue for the extended version.

Compute is not the constraint. Thirty-one A40s are visible on qTRDGPU and the
downstream grid is roughly 24 to 30 runs at 12 hours each, which is about two
and a half days of wall clock at six concurrent. The constraint is the
translator, so it is built first and a CycleGAN baseline is trained in parallel
as insurance.

| Days | Dates | Work |
|---|---|---|
| 1-2 | 8-9 Oct | Differentiable physics layer (done, verified). Refiner, discriminator, training loop. |
| 3-4 | 10-11 Oct | Translator trained. CycleGAN baseline in parallel. Lesion-preservation metric. |
| 5-6 | 12-13 Oct | Synthetic sets generated and audited. Qualitative figure. |
| 7-10 | 14-17 Oct | Downstream grid: budgets x arms x seeds. |
| 11-12 | 18-19 Oct | Ablations, failure analysis, final numbers frozen. |
| 13-18 | 20-25 Oct | Writing, Dr. Ye review cycles, revision. |
| 19 | 26 Oct | Submit. |

Risk control: the baseline curve is already running, the CycleGAN arm is
insurance against the proposed method failing, and every claim rests on
measurements that already exist if the new ones do not land.

## 8. Protocol caveat to carry into the paper

Validation stays the full 725-image target set at every budget, so the curve
isolates training-set size. The honest headline therefore quotes **total** label
cost, train budget plus 725. A budgeted-validation ablation is follow-up work.
A reviewer will raise this, so it is stated up front.

## 9. Running now

Label-efficiency baseline, budgets 0/250/500/1000/2000 at seed 0, jobs
4572237 to 4572241, via `scripts/71_label_efficiency_baseline.slurm`. Budget
3402 already exists at three seeds. This curve is the denominator for every
synthesis arm and is needed regardless of which translator wins.
