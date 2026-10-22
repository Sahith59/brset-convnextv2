# Stage 2 design spec: the learned refiner

Every decision below is tied to a measurement we already have, not to convention.

## What stage 2 must actually fix

Stage 1 (bounded physics) is exhausted: fitted by gradient descent it reaches a
91.8 percent statistic-distance reduction with spot and hole amplitude BOTH
pinned at their ceiling, and contrast and saturation still short. The measured
residual gap is three things:

| Residual gap | Evidence | Who fixes it |
|---|---|---|
| Hue (BRSET pink-red, mBRSET orange) | qualitative audit of the fitted panel | stage 2 |
| Framing (circle fills frame vs black bars) | BRSET mean aspect 1.287 and circle inset; mBRSET uniform 1600x1600 circle-fills | **NOT stage 2, see D1** |
| Residual contrast and saturation | stage 1 params pinned at ceiling, still short of target | stage 2 |

## D1. Framing is handled geometrically, never by the generator

The framing difference is deterministic geometry, not appearance. Asking a
generator to learn it invites it to warp or translate anatomy, which is exactly
the lesion-destroying failure the whole design is built to prevent.

Instead: detect the fundus circle, crop to its bounding box, resize to square.
Deterministic, inspectable, and provably cannot move a lesion relative to its
surrounding retina. Applied identically to both datasets so it is a
normalisation, not a learned transform.

## D2. The refiner output is a BOUNDED residual

    out = clamp( x_phys + eps * tanh( f(x_phys) ), 0, 1 )

with eps small (start 0.10). Rationale is direct evidence, not caution: when the
physics stage was left unbounded, the optimiser drove spot and hole amplitude to
2.98 and 1.95 on a [0,1] image, hitting the target statistics by painting large
blobs. Numerically excellent, visually destroyed. Bounding was load-bearing
there and the same argument applies with more force to a network with thousands
of free parameters. eps is a reported hyperparameter and is swept.

## D3. Loss is statistic matching plus feature-statistic matching, NOT a GAN first

Three terms:
1. **Statistic loss.** The five measured statistics, already differentiable in
   `scripts/72_physical_layer.py`, against the measured mBRSET targets.
2. **Style loss.** Gram-matrix matching of frozen encoder features against real
   mBRSET images. This is what captures hue and texture, which the five scalar
   statistics provably miss.
3. **Residual penalty.** L1 on the residual, so the network uses the smallest
   change that works and the physics stage keeps doing the heavy lifting.

Adversarial loss is an OPTIONAL fourth term added only after the above is
working. Rationale: a 17-day deadline and GAN mode collapse is the single most
common way this fails. Gram-based style transfer is stable and trains in hours.

## D4. Feature extractor is convnextv2_nano, which is cached locally

Checked: no VGG weights are in the torch hub cache, and `download.pytorch.org`
returns 403 from the login node. A download dependency would fail on compute
nodes mid-run. `models--timm--convnextv2_nano.fcmae_ft_in22k_in1k` IS cached, so
it is used frozen. Decision driven by verified availability, not preference.

## D5. Lesion preservation is MEASURED, not assumed

A translated BRSET image keeps its BRSET label, so a generator that erases a
microaneurysm silently corrupts the training set. The check: take a classifier
trained ONLY on real mBRSET (checkpoints exist: `mbrset_ft_lr2e5`,
`mbrset_scratch_stable`), run it on translated BRSET images, and measure
agreement with the known BRSET label. Reported per method and per eps, so the
appearance/fidelity trade-off is visible rather than claimed.

This is the methodological contribution. The generator backbone is commodity;
the guarantee is not.

## Open questions for the critique pass

- Is Gram matching on a frozen ImageNet encoder enough to move hue, or does hue
  need an explicit colour transform term?
- Does eps = 0.10 leave enough capacity to close the gap at all?
- Is agreement-with-source-label the right preservation metric, or does it just
  measure classifier robustness?
- What does a reviewer attack first?
