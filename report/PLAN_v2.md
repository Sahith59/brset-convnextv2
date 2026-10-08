# PLAN v2: Dual-domain device translation for cross-camera retinal screening

**Version:** 2.0 | **Date:** 8 October 2026 | **Status:** ACTIVE
**Target:** ISBI 2027, four-page paper, deadline **26 October 2026**
**Fallback:** MIDL 2027, paper deadline 4 December 2026

**Supersedes:**
- `report/RESEARCH_PLAN.md` (v1, architecture-side domain adaptation) - SUPERSEDED
- `report/PAPER_PLAN.md` (v1.5, first pivot draft, MIDL-targeted) - SUPERSEDED

Live status is tracked separately in `report/WORKING_MEMORY.md`.

---

## 1. The pivot, in one paragraph

For six weeks the project asked "what architecture closes the gap between a
hospital fundus camera and a handheld one." Five architectures were built and
all five failed. On 8 October a single measurement explained why: the gap was
already closed by the simplest possible method, so there was nothing left for
any architecture to win. The project now asks a different question, the one the
advisor proposed and the one a screening programme actually has:
**can synthetic handheld images replace real handheld labels?**

## 2. The measurement that forced the pivot

All at matched best-cutoff, same 732-image target test set, 159 DR positives.

| | DR F1 |
|---|---|
| Best of six models trained DIRECTLY on target data | 0.8456 |
| Mean of those six | 0.8271 |
| **Pooling BRSET + all 3,402 real target images** | **0.8455** |
| Pooling, 6-model ensemble | 0.8581 |

Pooling already equals the target-trained ceiling. The seed-only noise floor is
**0.0199 DR F1 and 9 patients**; every architecture tested landed inside it, and
even ensembling gains only 0.0126 and stays inside it. **The setting was
saturated.** This is now the paper's premise rather than its obstacle.

## 3. Why the data axis has headroom

| Dataset | Images | DR positives | Rate |
|---|---|---|---|
| BRSET train (tabletop, hospital) | 11,372 | 748 | 6.6% |
| mBRSET train (handheld, community) | 3,402 | 797 | 23.4% |

The 3,402 handheld images contribute MORE positives than the 11,372 hospital
images. They are also the expensive ones: hospital data already exists, handheld
data must be collected in the field and graded. Zero-shot (no handheld images at
all) scores DR F1 0.7842 with 50 of 159 missed, against pooling's 0.8455.

**That 0.06 F1 and ~20 patients is the headroom, and it lives on the data axis.**
The question is how far up that curve synthesis can carry us without real labels.

## 4. The method: dual-domain translation, after DuDoNet

Dr. Ye pointed to **DuDoNet** (Lin et al., CVPR 2019, arXiv 1907.00273), a CT
metal-artifact method. The relevance is structural. DuDoNet restores in two
domains at once, the sinogram domain where acquisition physics lives and the
image domain where appearance lives, joined by a differentiable physical
operator so both train end to end. Neither domain alone is sufficient.

**Both halves of that claim are already measured for our problem:**

1. **Physics alone is insufficient.** The nine-parameter acquisition chain fitted
   to measured target statistics cut normalised appearance distance from 5.13 to
   1.36, yet the outputs still do not look like real handheld images. Framing and
   hue dominate perceptually and are invisible to the five fitted statistics.
2. **The physical model is exhausted, not merely imperfect.** Made differentiable
   and fitted by gradient descent under physically plausible bounds, it reaches a
   91.8 percent distance reduction with spot and hole amplitude BOTH pinned at
   their ceiling while contrast and saturation remain short of target. It cannot
   go further without leaving physical plausibility.
3. **Unconstrained learning is unsafe.** Removing the bounds drives spot and hole
   amplitude to 2.98 and 1.95 on a [0,1] image: the optimiser hits the target
   statistics by painting large blobs. Numerically excellent, visually destroyed.

**Architecture.**

```
BRSET image (keeps its BRSET label)
    |
    v
 STAGE 1  PhysicalAcquisition          <- scripts/72, bounded, 9 interpretable
          blur, illumination, artifacts,   parameters, differentiable, cannot
          sensor noise, colour pipeline    move anatomy (photometric only)
    |
    v
 STAGE 2  Residual refiner             <- learned, bounded residual, closes
          (framing, hue, texture)         what physics provably cannot
    |
    v
 synthesised handheld image  -->  adversarial loss vs real mBRSET
                             -->  statistic-matching loss vs measured target
                             -->  structure/lesion preservation constraint
```

Trained jointly. The bounded physics stage is the structural prior that keeps
the generator honest.

**Why labels survive.** A translated BRSET image keeps its BRSET label, so a
generator that erases a microaneurysm silently corrupts the training set. Stage 1
is photometric and cannot move anatomy at all. Stage 2 is held to a bounded
residual on top of it. Preservation is then **measured**, not assumed, with a
classifier trained only on real target data.

## 5. Claims the paper makes

1. **Hand-crafted degradation is mis-specified for this device pair.** The gap is
   noise and contrast, not blur. GDRNet's published settings score appearance
   distance 7.75 against untouched BRSET's 5.13: worse than doing nothing.
   *Status: measured.*
2. **Physics-only translation saturates with the gap still open.** Bounded
   gradient fitting pins two parameters at their ceiling and still misses
   contrast and saturation. *Status: measured.*
3. **Dual-domain translation closes the residual while preserving lesions.**
   Verified on appearance distance AND a lesion-preservation score.
   *Status: to be run.*
4. **HEADLINE: synthetic target images substitute for real target labels.**
   Joint training with synthesis reaches full-pooling performance at K real
   labels, K far below 3,402. The claim is a curve. *Status: to be run.*
5. **Every comparison clears a measured noise floor.** 0.0199 DR F1, 9 patients.
   Saturation at full data is reported honestly and explains the negatives.
   *Status: measured.*

Claims 1, 2 and 5 are done. Claims 3 and 4 are the remaining work.

## 6. Experiment grid

**Axis A, real target label budget:** 0, 250, 500, 1000, 2000, 3402
**Axis B, what is added to BRSET:**

| Arm | Content | Priority |
|---|---|---|
| B0 | nothing (baseline curve) | running |
| B4 | **dual-domain translation (proposed)** | 1 |
| B3 | CycleGAN / CUT unpaired translation | 2 (insurance + expected baseline) |
| B2 | fitted hand-crafted chain | 3 (parameters already fitted) |
| B1 | GDRNet published settings | 4 (the negative control) |

**Seeds:** 2 per cell, 3 for headline cells. Nothing is reported that does not
clear 0.0199.

Not every cell is required. Order: B0 curve, then B4 and B3 at budgets
250/500/1000, then B2 and B1 at those budgets, then full-budget convergence cells.

## 7. Schedule to 26 October

| Days | Dates | Work |
|---|---|---|
| 1-2 | 8-9 Oct | Differentiable physics layer (DONE, verified). Refiner, discriminator, joint training loop. |
| 3-4 | 10-11 Oct | Translator trained. CycleGAN baseline in parallel. Lesion-preservation metric. |
| 5-6 | 12-13 Oct | Synthetic sets generated and audited. Qualitative figure. |
| 7-10 | 14-17 Oct | Downstream grid. |
| 11-12 | 18-19 Oct | Ablations, failure analysis, numbers frozen. |
| 13-18 | 20-25 Oct | Writing and advisor review cycles. |
| 19 | 26 Oct | Submit. |

Compute is not the constraint: 31 A40s visible, grid is ~24-30 runs at 12h, about
2.5 days wall clock at six concurrent. The translator is the critical path.

## 8. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Dual-domain translator fails to train | CycleGAN arm trains in parallel from day 3 |
| Synthesis helps but inside the noise floor | Report as a bounded negative; claims 1, 2, 5 still carry a 4-page paper |
| Lesion preservation fails | It is measured, so this becomes a reported finding, not a hidden flaw |
| Grid does not finish | Budgets 250/500/1000 are the headline; 0 and 2000 are supporting |

## 9. Protocol caveats to state in the paper

- Validation stays the full 725-image target set at every budget, so the curve
  isolates training-set size. The headline must therefore quote **total** label
  cost (budget + 725). A budgeted-validation ablation is named as future work.
- BRSET and mBRSET share no patients, so translation is unpaired by necessity.
- Images are credentialed under a PhysioNet DUA and are never redistributed.
