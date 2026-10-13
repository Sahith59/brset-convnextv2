# ISBI 2027 four-page outline

Hard limit 4 pages including figures, excluding references. Deadline 26 Oct 2026.
Every number cites a claim ID from `paper/CLAIMS.md`.

## Title (working)
Learned device translation for cross-camera retinal screening: how many real
handheld labels do you actually need?

## Abstract (~150 words)
Gap, saturation (C2), what we build, headline result (C11), one honest negative.

## 1. Introduction (~0.6 page)
- Screening matters most where the handheld camera is used; model trained on
  tabletop data misses 42 of 159 (C1).
- The obvious fix, pooling, works, but saturates (C2) and needs real labels that
  are expensive to collect in the field.
- So the question is label efficiency, not architecture.
- Contributions: (i) the gap is noise/contrast not blur (C5); (ii) a published
  degradation method moves the wrong way (C6); (iii) a dual-domain translator
  with a measured lesion-preservation guarantee (C9, C10); (iv) a label-efficiency
  curve showing synthesis substitutes for N real labels (C11).

## 2. Related work (~0.3 page, compressed)
cofe-Net and GDRNet hand-crafted degradation; CycleGAN SLO-to-CF for glaucoma
(names diffusion as future work); paired camera-to-cSLO Sci Rep 2026 with no
downstream task; EyeBench's call for diagnosis-based evaluation. Position: none
does unpaired device-class translation with a preservation guarantee evaluated
on downstream label budget.

## 3. Method (~1 page) -- FIGURE 1 here
- 3.1 What the gap actually is. Table of C5. One paragraph.
- 3.2 Acquisition stage. Nine interpretable parameters, differentiable, bounded.
  State C7: bounded fitting pins two parameters and still falls short; unbounded
  fitting paints blobs. This motivates stage 2 empirically.
- 3.3 Refinement stage. Bounded residual, statistic + style loss, geometric
  framing normalisation handled separately and why.
- 3.4 Lesion preservation, defined and measured.

## 4. Experiments (~1.6 pages) -- FIGURES 2, 3, TABLE 1
- Setup: datasets, splits, backbone, the noise floor (C8). Short.
- 4.1 Appearance matching. Table 1: distances for untouched / GDRNet / fitted /
  ours (C6, C9). Figure 2: qualitative panel.
- 4.2 Lesion preservation vs eps (C10).
- 4.3 Label efficiency. FIGURE 3, the money figure: baseline curve (C4) with the
  synthesis arms overlaid (C11). Noise floor shown as a band.
- 4.4 Controls. Class balance rejected (C3). Saturation reported honestly (C2).

## 5. Conclusion (~0.2 page)
What transfers, what does not, and the one thing we could not resolve.

---

# Figures

| # | Content | Status |
|---|---|---|
| 1 | Method diagram: BRSET -> physics stage -> refiner -> synthetic, with losses | to draw |
| 2 | Qualitative 4-column: BRSET / GDRNet / physics-only / ours / real mBRSET | partially exists (`results/degradation_examples.png`) |
| 3 | **Label-efficiency curve with noise-floor band** | baseline half exists tonight |
| T1 | Appearance distance + downstream F1/AUC per arm per budget | half exists |

# Writing order (do NOT wait for experiments)
1. Section 3 method, now. Stage 1 is built and C7 is measured.
2. Section 4.4 controls and Setup, now. C2, C3, C8 are done.
3. Section 1 intro, now. C1, C5, C6 are done.
4. Section 2 related work, now. Literature already reviewed.
5. Sections 4.1-4.3 as results land.
6. Abstract and conclusion last.

That is roughly 60 percent of the paper writable before the translator finishes.

# Open logistics
- ISBI 2027 LaTeX template not yet downloaded. Needed early, not on day 16.
- Decide author list and affiliations with the advisor.
