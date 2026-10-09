# Evidence ledger

Every claim the paper makes, its exact number, where that number comes from, and
how many seeds back it. Nothing enters the paper that is not in this file.
Status: MEASURED means the number exists today.

---

## C1. The cross-camera gap is real and large
**MEASURED.** Zero-shot BRSET -> mBRSET: DR F1 **0.7826**, **42 of 159** diseased
patients missed, AUC 0.9104. Against pooling at 0.8455. Test set 732 images,
159 DR+, 63 ME+, element-wise identical across every model compared.
Source: `results/lebase_b0_s0/`. Seeds: 1 now, 3 by tonight.

## C2. Pooling saturates the target-trained ceiling
**MEASURED.** Pooling BRSET + all 3,402 real target images: **0.8455** (3-seed
mean, best-cutoff). Best of six models trained DIRECTLY on target data:
**0.8456**. Mean of those six: 0.8271. Even a 6-model ensemble reaches only
0.8581, inside the noise floor.
Why it matters: this is why five architecture-side interventions all failed, and
it is the justification for moving to the data axis.
Source: verified 8 Oct, all eight test sets element-wise identical. Seeds: 3.

## C3. Class balance does NOT explain the pooling gain
**MEASURED.** BRSET train is 6.6% DR+, mBRSET 23.4%; pooling more than doubles
absolute positives, 748 -> 1545. Control: BRSET only, positives resampled to
exactly 23.4% (verified achieved 0.2340).

    budget 0, no rebalancing       0.7826
    budget 0, rebalanced to 23.4%  0.7843   (2 seeds: 0.7972, 0.7713)
    pooling, all 3402 real         0.8455

Class balance closes **+0.0017 of +0.0629**, three percent, far inside the
0.0199 floor. Even the most favourable seed (0.7972) is still 0.0483 short.
Why it matters: rules out the main alternative explanation a reviewer will
raise, and it was the advisor's own hypothesis. Seeds: 2, third running.

## C4. The curve plateaus at roughly 500 real target images, not 3,402
**MEASURED (seed 0; seeds 1-2 running).**

    budget      0    250    500   1000   2000   3402
    DR AUC  0.9104 0.9278 0.9422 0.9424 0.9453 0.9377
    DR F1   0.7826 0.7785 0.8167 0.8456 0.8350 0.8383

AUC rises to 0.9422 by budget 500 then flat. Steps clearing the floor:
250->500 (+0.0382), 500->1000 (+0.0289). Steps inside the floor: 1000->2000,
2000->3402. Caveat: 250 scores slightly BELOW 0 on F1 while missing fewer
patients, which is noise; this is why seeds 1-2 are running.
Why it matters: it sets the synthesis target precisely. Reach ~0.942 AUC with
fewer than 500 real labels.

## C5. The appearance gap is noise and contrast, NOT blur
**MEASURED, and CORRECTED DOWNWARD 9 Oct.** 400 images per dataset.

The earlier "5 to 7 times more high-frequency energy" was inflated by a scale
confound: the comparison varied output resolution but always applied a fixed
3x3 Laplacian at matched PIXEL pitch, never at matched retinal scale. BRSET's
fundus disc is 2491 px against mBRSET's 1599 px, so BRSET was being measured at
a finer retinal scale on a power-law spectrum.

Re-measured with the fundus disc resized to an identical 1024 px in both:

    sharpness ratio, native resolution      7.50x
    sharpness ratio, fundus-disc matched    4.05x   <- the honest number

The finding survives and the direction is unchanged: mBRSET carries far MORE
high-frequency energy, not less. But the paper must quote **4.05x**, not 5-7x.

Disc-matched, the rest of the gap:

    statistic    BRSET    mBRSET   ratio
    sharpness  0.00011   0.00046   4.05
    contrast   0.08623   0.14265   1.65
    brightness 0.39623   0.35926   0.91
    falloff    0.08711   0.09327   1.07
    saturation 0.80843   0.75780   0.94

Why it matters: the fundus-degradation literature models degradation as blur.
On this device pair that is the wrong axis, by a factor of four.

## C5b. The chromatic gap is a BLUE-CHANNEL gap. NEW, never previously measured.
**MEASURED 9 Oct**, identical under native and disc-matched measurement, so it
is not a scale artifact.

    channel fraction   BRSET    mBRSET   ratio
    red                0.59797  0.58318   0.98
    green              0.29678  0.26595   0.90
    blue               0.10525  0.15087   1.43   <-- dominant

    per-channel std    BRSET    mBRSET   ratio
    red                0.13579  0.13945   1.03
    green              0.09777  0.15604   1.60
    blue               0.05311  0.15183   2.86   <-- dominant

The handheld camera puts **43 percent more blue** into the image and **2.86
times the blue variation**, while red is essentially unchanged (1.03). The
qualitative "pink-red versus orange" impression is, quantitatively, a blue
channel difference.

Two consequences.
1. It explains why the previous 9-parameter acquisition model could not look
   right no matter how it was fitted: it had no per-channel degree of freedom,
   so it could not move blue independently of red. Verified separately: max
   achievable R:G:B ratio change was 0.00088.
2. The excess high-frequency energy is concentrated in green (1.60) and blue
   (2.86) and absent in red (1.03). That is the signature of small-sensor read
   and shot noise, not of focus, and it is why the acquisition model now uses
   signal-dependent noise rather than a single i.i.d. term.

## C6. GDRNet's published settings are worse than no augmentation
**MEASURED.** Normalised appearance distance to measured mBRSET:
untouched BRSET **5.13**, GDRNet published **7.75**, our fitted chain **1.36**.
GDRNet moves AWAY from the target on four of five statistics.
Why it matters: a direct, reproducible challenge to a MICCAI 2023 method.

## C7. The physical model is exhausted, not merely imperfect
**MEASURED.** Made differentiable and fitted by gradient descent under
physically plausible bounds: 91.8% statistic-distance reduction, with spot AND
hole amplitude both pinned at their ceiling while contrast and saturation remain
short. Unbounded, the optimiser drives those amplitudes to **2.98 and 1.95** on a
[0,1] image, hitting the statistics by painting blobs.
Why it matters: this is the empirical argument for the dual-domain design. The
learned stage is needed because physics provably cannot reach, not because
learning is fashionable.
Source: `scripts/72_physical_layer.py`, verified 8 Oct.

## C8. Every comparison clears a measured noise floor
**MEASURED.** Seed-only repetition of the identical configuration: DR F1 spread
**0.0199**, missed patients spread **9**, ME F1 spread 0.0424.
Why it matters: most papers in this space report single runs. This is the
methodological spine and it is what makes the negatives credible.

---

# NOT YET MEASURED (the remaining work)

## C9. Dual-domain translation closes the residual appearance gap
Needs: stage 2 built and trained. Metric: appearance distance below 1.36, plus
qualitative panel.

## C10. Translation preserves lesions
Needs: the preservation metric. Reported per method and per residual bound eps.

## C11. HEADLINE. Synthetic target images substitute for real target labels
Needs: synthesis arms of the curve at budgets 0/250/500, versus C4's baseline,
versus the hand-crafted chain, versus GDRNet, versus CycleGAN.
Success condition: at a given budget, synthesis beats no-synthesis by more than
0.0199, and beats the hand-crafted arm.
