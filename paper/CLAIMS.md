# Evidence ledger

Every claim the paper makes, its exact number, where that number comes from, and
how many seeds back it. Nothing enters the paper that is not in this file.

**THRESHOLD POLICY, set 9 Oct after review.** Two F1 numbers exist for every run.
The *val-threshold* number uses the cutoff tuned on the 725-image target
validation set, which is what a deployed system can actually do. The *oracle*
number sweeps the cutoff on the TEST set and is an upper bound no deployment can
reach. Earlier versions of this file silently quoted the oracle everywhere; at
budget 1000 that inflated F1 by +0.0323, which is larger than the 0.0199 noise
floor. **DR AUC is now the headline metric** because it needs no threshold at
all. Val-threshold F1 is secondary. Oracle numbers appear only where labelled.

## C1. The cross-camera gap is real and large
**MEASURED.** Zero target *training* labels, BRSET -> mBRSET:
**DR AUC 0.9104**, val-threshold DR F1 0.7774, 42 of 159 missed
(oracle F1 0.7826). Test set 732 images, 159 DR+, element-wise identical across
every model compared.

**Naming correction.** This is NOT "zero-shot". The run still consumes the
725-image target validation set for checkpoint selection (argmaxed over 25
epochs x 2 variants) and for threshold tuning. Honest label cost at this point
is **725 target labels**, and at budget 500 it is **1,225**. The paper says
"zero target training labels" and puts total label cost on the x-axis.
Source: `results/lebase_b0_s0/`. Seeds: 1 now, 3 landing tonight.

## C2. Pooling saturates the target-trained ceiling
**MEASURED, and it SURVIVES the threshold correction.** Recomputed 9 Oct at both
threshold policies:

                              val-threshold F1   oracle F1    DR AUC
    pooling, 3-seed mean               0.8365      0.8455    0.9463
    best of six target-trained         0.8355      0.8456    0.9446
    mean of six target-trained         0.8209      0.8271    0.9409
    difference (pooling minus best)   +0.0010     -0.0001   +0.0017

Pooling equals the best directly-target-trained model under every metric and
every threshold policy, and the difference is two orders of magnitude inside the
0.0199 noise floor. A 6-model ensemble reaches only 0.8581 oracle, also inside.

Why it matters: this is why five architecture-side interventions all failed, and
it is the justification for moving to the data axis. It is the one load-bearing
claim that did not weaken under review.
Source: 8 models, all test sets verified element-wise identical. Seeds: 3.

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

## C4. The curve plateaus, but ONLY on AUC. Corrected 9 Oct.
**MEASURED (seed 0; seeds 1-2 landing tonight).**

    budget          0      250      500     1000     2000     3402
    DR AUC     0.9104   0.9278   0.9422   0.9424   0.9453   0.9377
    val-thr F1 0.7774   0.7785   0.8014   0.8133   0.8317   0.8250
    oracle F1  0.7826   0.7785   0.8167   0.8456   0.8350   0.8383

**AUC plateaus at budget 500** (0.9422, then 0.9424, 0.9453, 0.9377: flat within
noise). **Val-threshold F1 does NOT plateau**; it rises monotonically through
2000. The earlier "plateaus at roughly 500" rested on oracle F1, where budget
1000 was inflated by +0.0323.

The honest statement: **the ranking signal saturates by 500 target images, while
the operating point keeps improving to 2000.** That distinction is itself worth
stating, since it means the residual benefit of more target data is calibration
rather than representation.

Caveat: single seed. Budget 250 scores below budget 0 on F1 while missing fewer
patients, which is noise. Seeds 1-2 running.

## C5. The appearance gap is noise and contrast, NOT blur
**UNDER RE-DERIVATION. DO NOT QUOTE A NUMBER YET.** Review raised a second
contamination: the Laplacian is computed on the full frame then masked, so the
field-of-view RIM (retina meeting black surround, a hard step) is included.
BRSET's disc is clipped by the frame top and bottom so part of its boundary is
the image border with no step, while mBRSET's disc is fully inscribed and has a
complete 360-degree rim. If the rim dominates, the statistic measures FOV
geometry, not image quality. `scripts/74_audit_physics.py` tests this by eroding
the mask; the ratio may fall to roughly 2.4x. This would be the THIRD version of
this number, so nothing is written until the audit lands.

Previous (now suspect) values below, retained for the record.
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
