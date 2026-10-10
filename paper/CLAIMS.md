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
**SETTLED 9 Oct, third and final version. The honest ratio is 2.1 to 2.5x.**

This number was wrong twice and the paper must carry the right one, so the
derivation is recorded in full.

    version 1   "5 to 7x"    inflated by a retinal-scale confound
    version 2   "4.05x"      disc-matched, but still rim-contaminated
    version 3   "2.1-2.5x"   rim excluded, and SCALE-INVARIANT

The contamination. image_stats computes a Laplacian on the full frame and then
masks, so pixels at the field-of-view rim, where retina meets black surround in
a hard step, are included. The two datasets have different FOV geometry: BRSET's
disc is clipped by the frame top and bottom, so part of its boundary is the
image border with no step, while mBRSET's disc is fully inscribed and carries a
complete 360-degree rim. Measured mask fractions confirm it: BRSET 0.833 against
mBRSET 0.775, where an inscribed circle gives pi/4 = 0.785.

Measured with the mask eroded, `scripts/74_audit_physics.py`:

    erode px        0      1      2      5     10     20
    native       5.82   2.74   2.19   2.13   2.12   2.13
    squash512    2.69   3.58   2.50   2.49   2.52   2.51

The ratio collapses once the rim is excluded and then goes flat. Critically it
also becomes **scale-invariant** (2.13 native against 2.51 at 512), which the
contaminated versions were not. That stability is the evidence that the eroded
measurement is the one tracking image content rather than frame geometry.

**What survives:** the direction, which is the part the paper needs. mBRSET
carries roughly 2.1 to 2.5 times MORE high-frequency energy than BRSET, not
less. The fundus-degradation literature models degradation as blur. On this
device pair that remains the wrong axis.

**What does not survive:** any claim of a 4x or 7x gap. The paper quotes 2.1-2.5x
and states the rim-exclusion protocol, because a reviewer who recomputes without
eroding will get a different number and needs to know why.

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


## C11. HEADLINE. Synthetic target images substitute for real target labels
Needs: synthesis arms of the curve at budgets 0/250/500, versus C4's baseline,
versus the hand-crafted chain, versus GDRNet, versus CycleGAN.
Success condition: at a given budget, synthesis beats no-synthesis by more than
0.0199, and beats the hand-crafted arm.

## C10. Lesion preservation, MEASURED. The metric works and it has teeth.
**MEASURED 9 Oct**, 160 BRSET images, `scripts/84_lesion_retention.py`,
`results/lesion_retention_sq512.json`. Classifier-free: red-lesion candidates
are detected on the inverted green channel with orientation-wise vessel
suppression plus black-tophat, then matched between original and translated.

**The metric is validated by its controls**, which is what makes it citable:

    condition              retention   spurious
    identity (null)            1.000      0.000
    gaussian blur sigma 1      0.202      0.033
    gaussian blur sigma 3      0.005      0.168
    chance (random match)      0.045          -

Perfect on identity, near-total loss under blur, far above chance. It measures
what it claims to measure.

**FINDING 1. GDRNet destroys half the lesion evidence.**

    numpy55_gdrnet_generic     0.465      0.324

GDRNet's published settings retain only **46.5 percent** of red-lesion
candidates. This converts C6 from an appearance-distance argument into a
clinical one: the method does not merely move away from the target
distribution, it erases the findings the task depends on.

**FINDING 2. Sensor noise is a spurious-lesion generator, and it is the
dominant harm in our own chain.** Ablating the acquisition model:

    stage1_photometric_only    0.851      0.004   <- clean
    stage1_no_noise            0.885      0.134
    stage1_no_artifacts        0.842      0.725
    stage1_fitted_full         0.829      0.724

Removing the artifact blobs changes spurious not at all (0.725 vs 0.724).
Removing the NOISE drops it from 0.724 to 0.134. Under the noise sweep,
spurious reaches 0.92 to 0.94, meaning **over nine in ten detected candidates
in the synthetic image are fabricated**. Since a translated image keeps its
source label, that is a label-noise generator pointed at the 93.4 percent of
BRSET that is disease-negative.

**FINDING 3. The artifact blobs are NOT harmful. A prediction, tested and
refuted.** Review predicted the blobs would fabricate lesion-scale structures.
Measured across amplitude 0.20 to 0.40 and sigma 0.015 to 0.055, retention
stays 0.941 to 0.994 and spurious 0.014 to 0.025. The concern was reasonable
and it is wrong.

**FINDING 4. Blurring destroys lesions, which retroactively condemns the
original model.** Sweeping the signed sharpness parameter:

    sharpen_k  -0.95   retention 0.222
    sharpen_k  -0.60   retention 0.396
    sharpen_k  -0.30   retention 0.661
    sharpen_k  +0.00   retention 0.964
    sharpen_k  +1.00   retention 0.955   spurious 0.671
    sharpen_k  +2.50   retention 0.931   spurious 0.808

The superseded nine-parameter model fitted `blur_sigma` to 0.907, squarely in
the destructive region. **It was discarding roughly 60 to 78 percent of
lesion candidates in order to match a sharpness statistic that synthetic noise
was supplying anyway.** Sharpening preserves lesions but fabricates them.

**THE FRONTIER. This is the paper's central object.** There is a measurable
tension between appearance matching and lesion preservation, and the two
failure modes are distinct and separately attributable: blur loses real
lesions, noise and over-sharpening invent false ones, and photometric
operations do neither. Any synthesis method for this task must be placed on
that frontier, and nobody currently reports it.


## C9. The two-stage translator, first trained result
**MEASURED 9 Oct.** 4000 steps, 7.7 min on one A40. Fit evaluated on a HELD-OUT
source split the translator never saw. `results/xlate_v2_noartifact/`.

**Controlled comparison, artifact amplitude capped vs free:**

    quantity                        free (v1)   capped (v2)
    weighted z^2, FITTED             0.336        0.414
    brightness z, NOT fitted         +0.34        -0.07
    r_mean error, NOT fitted          0.069        0.018
    hole_amp                      pinned 0.40    capped 0.02
    visible haemorrhage-like blobs   2-3/image     none

Fitted freely the model scores better on the objective and worse on everything
the objective does not contain. Dark blobs lower the mean and raise the standard
deviation at once, so they are the cheapest simultaneous fix for brightness
(z +0.34) and contrast (z -0.32). The images then carry two or three dark round
structures indistinguishable from blot haemorrhages, on source images that keep
their original label, 93.4 percent of which are disease-negative.

Capping the amplitude costs **+0.078 weighted z^2** and removes the mechanism.
The unfitted diagnostics improve sharply, which is the evidence that the
artifact route was overfitting the objective rather than modelling the device.

**Capped fit quality, held out, all 13 fitted statistics:** every |z| <= 0.32.
Chromatic fractions r -0.18, g +0.10, b +0.21. Per-channel Laplacian variance
r -0.15, g -0.09, b -0.08. The chromatic gap that motivated the whole design is
closed.

**Honest caveats.**
- Eight of nineteen parameters pin at a bound (sharpen_k, spot_amp, hole_amp,
  noise_a_b, noise_b_r, noise_b_g, off_g, off_b). The family is straining, so no
  claim is made about any individual fitted value.
- `sharpness` as a diagnostic remains short, 0.00037 against 0.00056.
- Qualitatively the synthetic images still lack the strong specular flash
  artifact visible at the frame edge of many real mBRSET images, and carry a
  faint bright rim the real images do not have.
- Stage 2 adds little visible change over stage 1 at these settings.

## C11. Downstream result, FIRST ATTEMPT: synthesis as REPLACEMENT hurts.
**MEASURED 10 Oct.** Zero real target training labels. Identical to the
baseline at the same budget in every respect except that training images pass
through the trained translator. Val and test are real target images, never
translated.

    arm                       DR AUC   val F1   oracle F1   missed
    baseline, no synthesis    0.9104   0.7774      0.7826       42
    synthesis, translate_p=1  0.8888   0.7278      0.7584       44
    delta                    -0.0216  -0.0496     -0.0242       +2

    noise floor 0.0199 DR F1; the F1 change is 2.5x the floor
    reference ceiling, pooling all 3402 real: AUC 0.9463, val F1 0.8365

**This is a real negative, not noise.** It is also, most likely, a design error
rather than a verdict on synthesis. `translate_p = 1.0` REPLACES every source
image with its synthetic version, so the model never sees a real image. The
prior work that reports gains from generative augmentation trains on synthetic
PLUS real; the CycleGAN SLO-to-colour-fundus glaucoma study that lifted AUC
0.89 to 0.94 is explicit about this.

Replacement plausibly trades one domain gap for another: instead of BRSET
against mBRSET, the classifier now faces OUR SYNTHETIC mBRSET against real
mBRSET. The two are known to differ qualitatively even though the 13 fitted
statistics match to |z| <= 0.32: we add a faint bright rim that real images
lack, and we do not reproduce the specular flash common at the frame edge of
real handheld captures.

`--translate_p` is added to test this directly. Runs at p = 0.5 and p = 0.25
use synthesis as augmentation alongside real images rather than as a
substitution. Whatever the outcome, **the replacement-vs-augmentation
distinction is itself a reportable result**, since no paper in this area states
which regime it used or what the difference costs.
